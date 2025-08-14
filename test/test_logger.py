from src.logger import setup_logger

def test_setup_logger():
    logger =setup_logger(debug=True)
    logger.info("Hello, world!")
    
if __name__ == "__main__":
    test_setup_logger()