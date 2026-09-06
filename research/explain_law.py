"""Does IR = IC x sqrt(breadth) actually hold? Simulate it and see.

We build a fake forecaster with a KNOWN, tiny skill level, let it make N
independent bets, and measure the information ratio it achieves. If the law is
real, the measured IR should land on IC x sqrt(N).
"""
import numpy as np

rng = np.random.default_rng(0)
YEARS, TRIALS = 1000, 1


def simulate(ic, n_bets_per_year, years=YEARS):
    """A forecaster whose predictions correlate with outcomes at exactly `ic`."""
    total = years * n_bets_per_year
    truth = rng.normal(size=total)
    noise = rng.normal(size=total)
    # Blend signal and noise so corr(forecast, truth) == ic exactly.
    forecast = ic * truth + np.sqrt(1 - ic**2) * noise
    # Bet in proportion to the forecast; profit is forecast x outcome.
    pnl = np.sign(forecast) * truth
    yearly = pnl.reshape(years, n_bets_per_year).sum(axis=1)
    return yearly.mean() / yearly.std()


print("A forecaster with IC = 0.03 (a realistic professional signal)")
print("is right about 51.5% of the time. Barely better than a coin.\n")
print(f"{'bets/year':>10} {'sqrt(bets)':>11} {'predicted IR':>13} {'measured IR':>12}")
for n in [32, 100, 300, 720, 3000, 37800]:
    pred = 0.03 * np.sqrt(n)
    meas = simulate(0.03, n)
    print(f"{n:>10} {np.sqrt(n):>11.1f} {pred:>13.2f} {meas:>12.2f}")

print("\n\nSame idea, held fixed at our 32 bets/year -- varying skill instead:")
print(f"{'IC':>8} {'right %':>9} {'IR':>8}")
for ic in [0.01, 0.03, 0.05, 0.10, 0.18]:
    pct = 50 + 100 * np.arcsin(ic) / np.pi  # P(correct sign) for a bivariate normal
    print(f"{ic:>8.2f} {pct:>8.1f}% {simulate(ic, 32):>8.2f}")
