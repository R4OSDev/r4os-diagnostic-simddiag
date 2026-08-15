const r4os = @import("r4os");

// SIMDD (0.56.30): SIMD-Verschraenkungstest fuer die Lazy-FPU-Abnahme.
// Zwei R4X-Threads iterieren Vektorketten (@Vector(8, u32), bei AVX2 in
// YMM-Registern) und schlafen zwischen den Runden, damit der Scheduler
// zwischen beiden (und den Kernel-Tasks) wechselt. Jeder State-Verlust
// im Save/Restore-Pfad macht die Endsumme bitfalsch.
// /DYNAMICTASKS faehrt zusaetzlich zwei vollstaendig gejointe/reapte Wellen.
// Die erste hinterlaesst einen vergifteten, aber gueltigen MXCSR-Control-
// State; die zweite muss sauber und mit einem getrennten Seed-Satz starten.

comptime {
    asm (r4os.r4x.entryAsm("simdd_main"));
}

const rounds: u32 = 400;
const Vec = @Vector(8, u32);
const dynamic_arg = "/DYNAMICTASKS";
const irq_stress_arg = "/IRQSTRESS";
const irq_stress_rounds: u32 = 20_000_000;
const irq_stress_seed: u32 = 0x4952_5146;
const dynamic_worker_count: usize = 64;
const dynamic_wave_count: usize = 2;
const dynamic_seed_salts = [dynamic_wave_count]u32{ 0x0000_0000, 0x6D2B_79F5 };
const dynamic_result_sentinel: u64 = 0xD15E_A5E0_D15E_A5E0;
const mxcsr_control_mask: u32 = 0x0000_FFC0;
const mxcsr_poison_bit: u32 = 0x0000_2000;

var results: [dynamic_worker_count]u64 = .{0} ** dynamic_worker_count;
var done_flags: [dynamic_worker_count]bool = .{false} ** dynamic_worker_count;
var dynamic_mxcsr_control: u32 = 0;

fn chainVector(seed: u32, sys: anytype, sleep_every: u32) u64 {
    var init: [8]u32 = undefined;
    var i: u32 = 0;
    while (i < 8) : (i += 1) init[i] = seed +% (i *% 0x9E37_79B9);
    var v: Vec = init;
    const mul: Vec = @splat(@as(u32, 1664525));
    const add: Vec = @splat(@as(u32, 1013904223));
    var round: u32 = 0;
    while (round < rounds) : (round += 1) {
        v = v *% mul +% add;
        v ^= @as(Vec, @splat(round));
        if (round % sleep_every == 0) sys.sleepTicks(1);
    }
    const out: [8]u32 = v;
    var acc: u64 = 0;
    i = 0;
    while (i < 8) : (i += 1) acc +%= out[i];
    return acc;
}

fn chainScalar(seed: u32) u64 {
    return chainScalarRounds(seed, rounds);
}

fn chainScalarRounds(seed: u32, round_count: u32) u64 {
    var vals: [8]u32 = undefined;
    var i: u32 = 0;
    while (i < 8) : (i += 1) vals[i] = seed +% (i *% 0x9E37_79B9);
    var round: u32 = 0;
    while (round < round_count) : (round += 1) {
        i = 0;
        while (i < 8) : (i += 1) {
            vals[i] = vals[i] *% 1664525 +% 1013904223;
            vals[i] ^= round;
        }
    }
    var acc: u64 = 0;
    i = 0;
    while (i < 8) : (i += 1) acc +%= vals[i];
    return acc;
}

// Deliberately contains no syscall or function call in the hot loop. The
// vector must remain live while timer preemption and external R4D IRQ
// handlers run, which is the condition the ordinary sleep-at-call-boundary
// interleave test could not cover.
fn chainVectorTight(seed: u32, round_count: u32) u64 {
    var init: [8]u32 = undefined;
    var i: u32 = 0;
    while (i < 8) : (i += 1) init[i] = seed +% (i *% 0x9E37_79B9);
    var v: Vec = init;
    const mul: Vec = @splat(@as(u32, 1664525));
    const add: Vec = @splat(@as(u32, 1013904223));
    var round: u32 = 0;
    while (round < round_count) : (round += 1) {
        v = v *% mul +% add;
        v ^= @as(Vec, @splat(round));
    }
    const out: [8]u32 = v;
    var acc: u64 = 0;
    i = 0;
    while (i < 8) : (i += 1) acc +%= out[i];
    return acc;
}

var global_raw: ?*const r4os.abi.R4XStartContext = null;

fn workerMain(arg: u64) callconv(.c) i32 {
    const raw = global_raw orelse return 1;
    const start = r4os.r4xstart.Context.init(raw);
    var sys = start.r4sys() orelse return 1;
    const idx: usize = @intCast(arg);
    if (idx >= dynamic_worker_count) return 2;
    const seed = seedFor(idx);
    // Unterschiedliche Schlafraster => versetzte Switch-Punkte.
    const sleep_every: u32 = 2 + @as(u32, @intCast(idx % 7));
    @as(*volatile u64, &results[idx]).* = chainVector(seed, &sys, sleep_every);
    @as(*volatile bool, &done_flags[idx]).* = true;
    return 0;
}

export fn simdd_main(raw: *const r4os.abi.R4XStartContext) callconv(.c) i32 {
    const start = r4os.r4xstart.Context.init(raw);
    if (!start.valid()) return 513;
    var sys_val = start.r4sys() orelse return 514;
    const sys = &sys_val;
    global_raw = raw;

    sys.println("SIMDD SIMD interleave selftest");
    if (!sys.hasFn("thread_create_handle") or !sys.hasFn("thread_handle_join")) {
        sys.println("SIMDD result: SKIP (threads unsupported)");
        return 0;
    }
    if (argsEqual(start.args(), dynamic_arg)) return dynamicSimdProfile(sys);
    if (argsEqual(start.args(), irq_stress_arg)) return irqStressProfile(sys);

    var t0: r4os.abi.ProgramJoinHandle = .{};
    if (sys.threadCreateHandle(workerMain, 0, 0, 0, &t0) != r4os.abi.thread_ok) {
        sys.println("SIMDD result: FAILED thread0 create");
        return 1;
    }
    var t1: r4os.abi.ProgramJoinHandle = .{};
    if (sys.threadCreateHandle(workerMain, 1, 0, 0, &t1) != r4os.abi.thread_ok) {
        var ignored_exit: i32 = 0;
        _ = sys.threadHandleJoin(&t0, r4os.abi.thread_wait_forever, &ignored_exit);
        sys.println("SIMDD result: FAILED thread1 create");
        return 1;
    }

    // Der Hauptthread rechnet parallel eine dritte Kette - noch mehr
    // verschraenkte Switches zwischen drei FPU-Nutzern.
    const main_result = chainVector(0x0BAD_F00D, sys, 4);

    var exit0: i32 = 0;
    var exit1: i32 = 0;
    if (sys.threadHandleJoin(&t0, r4os.abi.thread_wait_forever, &exit0) != r4os.abi.thread_ok) {
        sys.println("SIMDD result: FAILED join0");
        return 1;
    }
    if (sys.threadHandleJoin(&t1, r4os.abi.thread_wait_forever, &exit1) != r4os.abi.thread_ok) {
        sys.println("SIMDD result: FAILED join1");
        return 1;
    }

    var ok = exit0 == 0 and exit1 == 0 and
        @as(*volatile bool, &done_flags[0]).* and
        @as(*volatile bool, &done_flags[1]).*;
    if (@as(*volatile u64, &results[0]).* != chainScalar(0x1234_5678)) ok = false;
    if (@as(*volatile u64, &results[1]).* != chainScalar(0x8765_4321)) ok = false;
    if (main_result != chainScalar(0x0BAD_F00D)) ok = false;

    if (!ok) {
        sys.println("SIMDD result: FAILED interleave mismatch");
        return 1;
    }
    sys.write("SIMDD result: OK rounds=");
    sys.printU64(rounds);
    sys.write(" threads=3 vec=8x u32");
    sys.println("");
    return 0;
}

fn irqStressProfile(sys: *r4os.r4xstart.R4Sys) i32 {
    sys.write("SIMDD irqstress start rounds=");
    sys.printU64(irq_stress_rounds);
    sys.println(" hotloop=yes");

    const actual = chainVectorTight(irq_stress_seed, irq_stress_rounds);
    const expected = chainScalarRounds(irq_stress_seed, irq_stress_rounds);
    if (actual != expected) {
        sys.write("SIMDD irqstress result: FAILED actual=");
        sys.printU64(actual);
        sys.write(" expected=");
        sys.printU64(expected);
        sys.println("");
        return 1;
    }
    sys.write("SIMDD irqstress result: OK rounds=");
    sys.printU64(irq_stress_rounds);
    sys.println(" vec=8x u32 liveAcrossIrq=OK");
    return 0;
}

fn dynamicSimdProfile(sys: *r4os.r4xstart.R4Sys) i32 {
    @as(*volatile u32, &dynamic_mxcsr_control).* = readMxcsr() & mxcsr_control_mask;
    var first_wave_results: [dynamic_worker_count]u64 = .{0} ** dynamic_worker_count;

    if (!runDynamicWave(sys, 0, null)) return 1;
    for (results, 0..) |result, index| first_wave_results[index] = result;
    if (!runDynamicWave(sys, 1, &first_wave_results)) return 1;

    sys.write("SIMDD dynamic result: OK waves=");
    sys.printU64(dynamic_wave_count);
    sys.write(" workers=");
    sys.printU64(dynamic_worker_count);
    sys.write(" active=65 rounds=");
    sys.printU64(rounds);
    sys.println(" avx2Isolation=OK freshInit=OK reuse=OK");
    return 0;
}

fn runDynamicWave(sys: *r4os.r4xstart.R4Sys, wave: usize, previous_results: ?*const [dynamic_worker_count]u64) bool {
    if (wave >= dynamic_wave_count) {
        sys.println("SIMDD dynamic result: FAILED invalid wave");
        return false;
    }
    @memset(results[0..], dynamic_result_sentinel);
    @memset(done_flags[0..], false);
    var handles: [dynamic_worker_count]r4os.abi.ProgramJoinHandle = .{r4os.abi.ProgramJoinHandle{}} ** dynamic_worker_count;
    defer {
        for (&handles) |*handle| {
            if (!validJoinHandle(handle.*)) continue;
            var ignored_exit: i32 = 0;
            _ = sys.threadHandleJoin(handle, r4os.abi.thread_wait_forever, &ignored_exit);
            handle.* = .{};
        }
    }
    var ids: [dynamic_worker_count]u32 = .{0} ** dynamic_worker_count;
    var index: usize = 0;
    while (index < ids.len) : (index += 1) {
        if (sys.threadCreateHandle(dynamicWorkerMain, encodeDynamicWorkerArg(wave, index), 0, 0, &handles[index]) != r4os.abi.thread_ok or !validJoinHandle(handles[index])) {
            sys.println("SIMDD dynamic result: FAILED create");
            return false;
        }
        ids[index] = handles[index].thread_id;
        if ((index + 1) % 16 == 0) {
            sys.write("SIMDD dynamic create wave=");
            sys.printU64(wave);
            sys.write(" progress=");
            sys.printU64(index + 1);
            sys.println("");
        }
    }

    const main_seed = mainSeedForWave(wave);
    const main_result = chainVector(main_seed, sys, 3);
    index = 0;
    while (index < ids.len) : (index += 1) {
        var exit_code: i32 = -1;
        if (sys.threadHandleJoin(&handles[index], r4os.abi.thread_wait_forever, &exit_code) != r4os.abi.thread_ok or exit_code != 0) {
            sys.println("SIMDD dynamic result: FAILED join");
            return false;
        }
        handles[index] = .{};
        var reaped_info: r4os.abi.ProgramThreadInfo = .{};
        if (sys.threadStatus(ids[index], &reaped_info) != r4os.abi.thread_error_not_found) {
            sys.println("SIMDD dynamic result: FAILED reap");
            return false;
        }
        const actual = @as(*volatile u64, &results[index]).*;
        if (!@as(*volatile bool, &done_flags[index]).* or
            actual == dynamic_result_sentinel or
            actual != chainScalar(seedForWave(wave, index)))
        {
            sys.println("SIMDD dynamic result: FAILED isolation");
            return false;
        }
        if (previous_results) |previous| {
            if (actual == previous[index]) {
                sys.println("SIMDD dynamic result: FAILED seed separation");
                return false;
            }
        }
        if ((index + 1) % 16 == 0) {
            sys.write("SIMDD dynamic join wave=");
            sys.printU64(wave);
            sys.write(" progress=");
            sys.printU64(index + 1);
            sys.println("");
        }
    }
    if (main_result != chainScalar(main_seed)) {
        sys.println("SIMDD dynamic result: FAILED main isolation");
        return false;
    }
    sys.write("SIMDD dynamic wave=");
    sys.printU64(wave);
    sys.write(" joined=");
    sys.printU64(dynamic_worker_count);
    sys.write(" reaped=");
    sys.printU64(dynamic_worker_count);
    sys.println(" freshInit=OK isolation=OK");
    return true;
}

fn validJoinHandle(handle: r4os.abi.ProgramJoinHandle) bool {
    return handle.thread_id != 0 and
        handle.instance_id != 0 and
        handle.thread_generation != 0 and
        handle.instance_generation != 0 and
        handle.reserved == 0;
}

fn dynamicWorkerMain(arg: u64) callconv(.c) i32 {
    const raw = global_raw orelse return 1;
    const start = r4os.r4xstart.Context.init(raw);
    var sys = start.r4sys() orelse return 1;
    const wave: usize = @intCast(arg >> 32);
    const index: usize = @intCast(arg & 0xFFFF_FFFF);
    if (wave >= dynamic_wave_count or index >= dynamic_worker_count) return 2;

    const expected_control = @as(*volatile u32, &dynamic_mxcsr_control).*;
    if (readMxcsr() & mxcsr_control_mask != expected_control) return 3;

    const sleep_every: u32 = 2 + @as(u32, @intCast(index % 7));
    @as(*volatile u64, &results[index]).* = chainVector(seedForWave(wave, index), &sys, sleep_every);
    @as(*volatile bool, &done_flags[index]).* = true;

    // Wave 0 leaves a valid but deliberately different rounding mode in the
    // task state. After every worker has been joined and reaped, wave 1 must
    // nevertheless start with the clean initial MXCSR control state.
    if (wave == 0) {
        const current = readMxcsr();
        const poisoned_control = expected_control ^ mxcsr_poison_bit;
        writeMxcsr((current & ~mxcsr_control_mask) | poisoned_control);
        sys.threadExit(0);
    }
    return 0;
}

fn encodeDynamicWorkerArg(wave: usize, index: usize) u64 {
    return (@as(u64, @intCast(wave)) << 32) | @as(u64, @intCast(index));
}

fn seedForWave(wave: usize, index: usize) u32 {
    return seedFor(index) ^ dynamic_seed_salts[wave];
}

fn mainSeedForWave(wave: usize) u32 {
    return 0x0BAD_F00D ^ dynamic_seed_salts[wave];
}

inline fn readMxcsr() u32 {
    var value: u32 = 0;
    asm volatile ("stmxcsr (%[dst])"
        :
        : [dst] "r" (&value),
        : .{ .memory = true });
    return value;
}

inline fn writeMxcsr(value: u32) void {
    var stored = value;
    asm volatile ("ldmxcsr (%[src])"
        :
        : [src] "r" (&stored),
        : .{ .memory = true });
}

fn seedFor(index: usize) u32 {
    return switch (index) {
        0 => 0x1234_5678,
        1 => 0x8765_4321,
        else => 0xA5A5_0001 +% (@as(u32, @intCast(index)) *% 0x1F12_3BB5),
    };
}

fn argsEqual(args: []const u8, expected: []const u8) bool {
    if (args.len != expected.len) return false;
    for (args, expected) |a, b| {
        const folded = if (a >= 'a' and a <= 'z') a - ('a' - 'A') else a;
        if (folded != b) return false;
    }
    return true;
}
