from setuptools import setup, find_packages

setup(
    name='rfcip',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "requests",
        "streamlit",
        "openpyxl",
        "beautifulsoup4",
        "tqdm",
        "altair",
    ],
    entry_points={
        "console_scripts": [
            "rfcip-app=streamlit_app:main",  # Optional CLI launcher
        ],
    },
)
