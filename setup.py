from setuptools import setup, find_packages

HYPEN_E_DOT = "-e ."
def getRequirements() -> list[str]:
    with open("requirements.txt", "r") as file:
        requirements = file.read().split("\n")
    requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name = "GemFit",
    version = "0.1",
    author = "Rauhan Ahmed Siddiqui",
    author_email = "rauhaan.siddiqui@gmail.com",
    packages = find_packages(where = "."), 
    requires = getRequirements()
)