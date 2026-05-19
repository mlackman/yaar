import pathlib
import tempfile 

import pytest

from ..tools import write_to_file, insert_to_file, read_file

class TestInsertToFile:

    @pytest.mark.asyncio
    async def test_insert_to_first_line(self):
        with tempfile.TemporaryDirectory() as temp:
            filename = str(pathlib.Path(temp) / 'hello-world.txt')
            data = "Hello\nWorld!"
            await write_to_file(filename, data)

            await insert_to_file(filename, line_no=1, data='jee ')
            assert 'jee Hello\nWorld!' == await read_file(filename)

    @pytest.mark.asyncio
    async def test_insert_to_second_line(self):
        with tempfile.TemporaryDirectory() as temp:
            filename = str(pathlib.Path(temp) / 'hello-world.txt')
            data = "First\nSecond"
            await write_to_file(filename, data)

            await insert_to_file(filename, line_no=2, data='insert this\n')
            assert 'First\ninsert this\nSecond' == await read_file(filename)

    @pytest.mark.asyncio
    async def test_append(self):
        with tempfile.TemporaryDirectory() as temp:
            filename = str(pathlib.Path(temp) / 'hello-world.txt')
            data = "First\nSecond"
            await write_to_file(filename, data)

            await insert_to_file(filename, line_no=-1, data='\ninsert this')
            assert 'First\nSecond\ninsert this' == await read_file(filename)
