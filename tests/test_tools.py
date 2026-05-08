import pytest
from src.tools import calculator, web_search, code_exec, file_io
from src.tools.executor import is_critical, execute_tool


def test_calculator():
    result = calculator.invoke({"expression": "2 + 3 * 4"})
    assert "14" in result


def test_calculator_unsafe():
    result = calculator.invoke({"expression": "__import__('os').system('ls')"})
    assert "izin verilmiyor" in result or "Hata" in result


def test_is_critical_code_exec():
    assert is_critical({"name": "code_exec", "args": {"language": "python", "code": "1+1"}})


def test_is_critical_file_io_read():
    assert not is_critical({"name": "file_io", "args": {"action": "read", "file_path": "test.txt"}})


def test_is_critical_file_io_write():
    assert is_critical({"name": "file_io", "args": {"action": "write", "file_path": "test.txt", "content": "x"}})


def test_execute_tool_calculator():
    result = execute_tool({"name": "calculator", "args": {"expression": "5 * 5"}})
    assert "25" in str(result)
