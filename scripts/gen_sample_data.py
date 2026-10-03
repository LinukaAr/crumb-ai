"""Generate synthetic sample bakery sales CSV with specific anomaly dates."""
import random

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

items = ['Croissant', 'Pain au chocolat', 'Sourdough Loaf', 'Cinnamon Roll', 'Espresso']
prices = {
    'Croissant': 280, 'Pain au chocolat': 320,
    'Sourdough Loaf': 850, 'Cinnamon Roll': 250, 'Espresso': 450
}

# Base daily demand per item (Mon-Sat patterns; shop closed Sundays)
base = {
    'Croissant':         [40, 38, 42, 45, 55, 70],
    'Pain au chocolat':  [25, 24, 26, 28, 35, 45],
    'Sourdough Loaf':    [12, 10, 11, 13, 15, 20],
    'Cinnamon Roll':     [20, 18, 22, 25, 30, 38],
    'Espresso':          [60, 55, 58, 62, 70, 85],
}

CLOSED_DAY = '2026-06-09'
PROMO_DAYS = {'2026-07-18', '2026-07-19'}
SLOW_DAYS  = {'2026-08-10', '2026-08-11', '2026-08-12'}

start = pd.Timestamp('2026-01-01')
end   = pd.Timestamp('2026-09-30')
dates = pd.date_range(start, end, freq='D')

rows = []
for date in dates:
    dow = date.dayofweek  # 0=Mon..6=Sun
    if dow == 6:
        continue  # closed Sundays

    date_str = date.strftime('%Y-%m-%d')
    is_holiday = 0
    rainy = int(random.random() < 0.2)
    promo = 0

    if date_str == CLOSED_DAY:
        for item in items:
            rows.append({
                'date': date_str, 'item': item, 'units_sold': 0,
                'unit_price_lkr': prices[item], 'day_of_week': dow,
                'is_holiday': 0, 'rainy': 0, 'promo': 0
            })
        continue

    if date_str in PROMO_DAYS:
        promo = 1

    if date_str in SLOW_DAYS:
        rainy = 1

    for item in items:
        b = base[item][min(dow, 5)]
        noise = float(np.random.normal(0, b * 0.15))
        rain_effect = -b * 0.3 if rainy else 0.0
        promo_effect = b * 0.6 if promo else 0.0
        units = max(0, round(b + noise + rain_effect + promo_effect))
        rows.append({
            'date': date_str, 'item': item, 'units_sold': units,
            'unit_price_lkr': prices[item], 'day_of_week': dow,
            'is_holiday': is_holiday, 'rainy': rainy, 'promo': promo
        })

df = pd.DataFrame(rows)
df.to_csv('data/sample_bakery_sales.csv', index=False)
n_dates = df['date'].nunique()
print(f'Generated {len(df)} rows, {n_dates} dates')

for check_date in ['2026-06-09', '2026-07-18', '2026-07-19', '2026-08-10', '2026-08-11', '2026-08-12']:
    d = df[df['date'] == check_date]
    if len(d):
        total = d['units_sold'].sum()
        pr = d['promo'].max()
        rn = d['rainy'].max()
        print(f'  {check_date}: {len(d)} rows, total={total}, promo={pr}, rainy={rn}')
    else:
        print(f'  {check_date}: MISSING')
