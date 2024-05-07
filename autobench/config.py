from os.path import abspath, dirname, expanduser, join

BASE_DIR = abspath(join(dirname(__file__), '../'))
DATA_DIR = join(BASE_DIR, 'data')
BGOS_DIR = join(BASE_DIR, 'bgo')

CONFIG_FILE = join(BASE_DIR, 'config.json')

INCLUDES = [
    expanduser('~/.local/include'),
    expanduser('~/graph-optimizer/include'),
]
