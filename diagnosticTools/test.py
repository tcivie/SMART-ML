from sim_database_converter import *

converter = SimJsonConverter(input("Enter file path:"))

converter.convert(export_to_json=True)
