import sys,os,pkgutil,re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def underscore_to_camel_case(name: str) -> str:
    """Convert snake_case to CamelCase."""
    return ''.join(word.capitalize() for word in name.split('_'))

def load_games_in_folder(folder:str)->list:
    sys.path.insert(0, folder)
    gameClasses=[]
    for _, module_name, _ in pkgutil.iter_modules([folder]):
        module = __import__(f'{module_name}', fromlist=[module_name])
        possibleNames = [module_name[0].upper()+module_name[1:], module_name.capitalize(), underscore_to_camel_case(module_name)]
        possibleNames.extend([name+'Game' for name in possibleNames])
        for name in possibleNames:
            if hasattr(module, name):
                gameClass=getattr(module, name)
                gameClasses.append(gameClass)
                break
    return gameClasses
