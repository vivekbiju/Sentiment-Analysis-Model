from src.data_processor import TextCleaner

def test_sanitization_logic(): # Mu
    cleaner = TextCleaner()
    result = cleaner.sanitize("<h1>Hello</h1>")
    assert result == "Hello"