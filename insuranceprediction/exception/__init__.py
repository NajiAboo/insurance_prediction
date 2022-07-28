from distutils.log import error
import os
import sys

class InsurancePredictionException(Exception):
    def __init__(self, error_meesage: Exception, error_details: sys) -> None:
        super().__init__(error_meesage)
        self.error_message = InsurancePredictionException.get_detailed_error_message(
                                                    error_meesage, error_details
                                                    )

    
    @staticmethod
    def get_detailed_error_message(error_message: Exception, error_details: sys) -> str:
        """
            Description:
            This function will return formatted custom exception

            Args:
                error_message : Exception object
                error_details : sys object
            Returns:
                formatted error message
            Raises:
                None
        """

        _,_ ,exec_tb = error_details.exc_info()
        exception_block_line_number = exec_tb.tb_frame.f_lineno
        try_block_line_number = exec_tb.tb_lineno
        file_name = exec_tb.tb_frame.f_code.co_filename

        error_message = f"""\
        Error occured in script:
            [{file_name}] at 
            try block line number: [{try_block_line_number}] and exception block line number: [{exception_block_line_number}]
            error message : [{error_message}]
        """

        return error_message

    
    def __str__(self) -> str:
        return self.error_message

    def __repr__(self) -> str:
        return InsurancePredictionException.__name__.str()