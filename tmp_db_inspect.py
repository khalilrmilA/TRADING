import sqlite3

path = 'data/trading.db'
conn = sqlite3.connect(path)
conn.row_factory = sqlite3.Row
c = conn.cursor()
print('TABLES:')
for row in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
    print(row['name'])
print('\nBOT_ACTIVITY SCHEMA:')
for row in c.execute("PRAGMA table_info(bot_activity)"):
    print(dict(row))
print('\nSAMPLE BOT_ACTIVITY:')
for row in c.execute("SELECT ts, symbol, action, detail FROM bot_activity ORDER BY ts DESC LIMIT 20"):
    print(row['ts'], row['symbol'], row['action'], row['detail'][:200])
print('\nOPEN POSITIONS:')
for row in c.execute("SELECT id, symbol, side, qty, entry_price, stop_loss, take_profit, status, pnl FROM paper_positions WHERE status='open' ORDER BY id"):
    print(dict(row))
print('\nRECENT CLOSED POSITIONS:')
for row in c.execute("SELECT id, symbol, side, qty, entry_price, exit_price, stop_loss, take_profit, pnl, exit_reason, closed_at FROM paper_positions WHERE status='closed' ORDER BY closed_at DESC LIMIT 20"):
    print(dict(row))
conn.close()
