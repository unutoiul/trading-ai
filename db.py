import asyncpg
import asyncio

CONNECTION = "postgres://tsdbadmin:lnf4snap4ajpp402@wl6tot2req.pv35kn44o5.tsdb.cloud.timescale.com:39212/tsdb?sslmode=require"


# async def main():
#     conn = await asyncpg.connect(CONNECTION)
#     extensions = await conn.fetch("select extname, extversion from pg_extension")
#     for extension in extensions:
#         print(extension)
#     await conn.close()

# asyncio.run(main())




async def main():
    conn = await asyncpg.connect(CONNECTION)
    
    # Query historical BTC/USDT data
    query = """
    SELECT time, open, high, low, close, volume
    FROM crypto_prices
    WHERE symbol = 'BTC/USDT'
    ORDER BY time DESC
    LIMIT 100;
    """
    
    rows = await conn.fetch(query)
    for row in rows:
        print(f"Date: {row['time']}, Open: {row['open']}, High: {row['high']}, Low: {row['low']}, Close: {row['close']}, Volume: {row['volume']}")
    
    await conn.close()

asyncio.run(main())