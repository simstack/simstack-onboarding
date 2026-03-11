import asyncio

import numpy as np
from simstack.core.context import context
from simstack.models.array_storage import ArrayStorage


async def make_numpy_array():
    await context.initialize()
    np_array = np.array([1, 2, 3])
    array_storage = ArrayStorage(name="test_array")
    array_storage.set_array(np_array)
    await context.db.engine.save(array_storage)


if __name__ == "__main__":
    asyncio.run(make_numpy_array())