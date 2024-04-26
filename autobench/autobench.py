import os
import sys
import json


from util import read_header, generate_code


signature = read_header('bgo/info/info/info.hpp')[0]

code = generate_code('bgo/info/info/info.hpp', signature)
print(code)
