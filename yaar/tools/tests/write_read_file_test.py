import pathlib
import tempfile 

import pytest

from ..tools import write_to_file, read_file

@pytest.mark.asyncio
async def test_read_write_file():
    with tempfile.TemporaryDirectory() as temp:
        filename = str(pathlib.Path(temp) / 'hello-world.txt')
        data = "Hello\nWorld!"

        await write_to_file(filename, data)

        read_data = await read_file(filename)

        assert data == read_data
