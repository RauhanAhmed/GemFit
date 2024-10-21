import sys

def error_message_detail(error):
    """
    Get exception information
    """
    _, _, exc_info = sys.exc_info()  
    filename = exc_info.tb_frame.f_code.co_filename  
    lineno = exc_info.tb_lineno 
    error_message = "Error encountered in line no [{}], filename : [{}], saying [{}]".format(lineno, filename, error)
    return error_message 

class CustomException(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)  # Call the parent class constructor
        self.error_message = error_message_detail(error_message) 

    def __str__(self) -> str:
        return self.error_message 
