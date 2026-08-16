SIMDD - SIMD-Verschraenkungstest (0.56.30 Lazy-FPU-Abnahme)

Zwei R4X-Threads rechnen verschraenkt AVX2-taugliche Vektorketten
(@Vector(8,u32)) ueber viele Task-Switches (sleepTicks zwischen den
Runden) und pruefen die Ergebnisse bitgenau gegen eine skalare
Referenz. Deckt FPU-/SIMD-State-Verlust im Scheduler auf (Lazy-FPU:
Kernel-Tasks ueberspringen Save/Restore, R4X-Tasks nicht).

Aufruf: SIMDD.R4X (AUTOEXEC-Smoke); Ausgabe "SIMDD result: OK ..."
