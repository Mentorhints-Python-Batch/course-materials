# Synchronous (Blocking)
import time
import asyncio

# def sync_task():
#     print("Task started")
#     time.sleep(2)  # Blocks entire program
#     print("Task completed")

async def async_task():
    print("Task started")
    await asyncio.sleep(2)  # Allows other tasks to run
    print("Task completed")

print()
print()
# sync_task
asyncio.run(async_task())
