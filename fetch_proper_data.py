from io import TextIOWrapper
import os

STATIONS_CODES = {
    "351150400": "Zielona_gora",
    "351160418": "Leszno",
    "351180435": "Kalisz",
    "351180455": "Wielun",
    "350180540": "Raciborz",
    "350150500": "Jelenia_gora",
    "350160520": "Klodzko",
    "350170530": "Opole",
    "351160415": "Legnica",
    "351160424": "Wroclaw-strachowice",
    "351160425": "Wroclaw"
}

PARAMETERS = {
    "B00202A": "kier_wiatr",
    "B00300S": "temp_pow",
    "B00305A": "temp_grunt",
    "B00604S": "suma_opad_doba",
    "B00702A": "sr_pred_wiatr",
    "B00802A": "wilg",
}

class Row:
    def __init__(self):
        for value in PARAMETERS.values():
            setattr(self, value, "null")
            
    def prepare_line(self):
        return ";".join([getattr(self, value) for value in PARAMETERS.values()])
        
def init_file_handlers(data_type: str):
    file_handlers = {
        key: open(os.path.join(".", data_type, f"{value}.csv"), "w", encoding="utf-8") 
        for key, value in STATIONS_CODES.items()
    }
    header = "date;" + ";".join(PARAMETERS.values()) + ";\n"
    for file_handler in file_handlers.values():
        file_handler.write(header)
    return file_handlers

def close_file_handlers(file_handlers: dict[str, TextIOWrapper]):
    for file_handler in file_handlers.values():
        file_handler.close()

def parse_line(line: str) -> tuple[str, str] | None:
    if not line:
        return None
    station_code, parameter, date, value, *_ = line.split(";")
    station_code = station_code.lstrip("\ufeff")
    if station_code not in STATIONS_CODES:
        return None
    if date[-5:] != "06:00":
        return None
    value = value.strip("\n")
    
    return station_code, date, value

def create_date_dict() -> dict[str, Row]:
    return {
        station_code: Row()
        for station_code in STATIONS_CODES
    }

def process_csv_files():    
    file_handlers = init_file_handlers("train_data")
    
    for root, _, files in os.walk(".\\data"):
        dates = {}
        for file in files:
            filepath = os.path.join(root, file)
            with open(filepath, mode="r", encoding="utf-8") as csv:
                line = True
                while line:
                    line = csv.readline()
                    result = parse_line(line)
                    if result is None:
                        continue
                    station_code, date, value = result
                    if dates.get(date) is None:
                        dates[date] = create_date_dict()
                    parameter = file.split("_")[0]
                    row = dates[date][station_code]
                    setattr(row, PARAMETERS[parameter], value)
        for date, stations in dates.items():
            stations: dict[str, Row]
            if date == "2021-09-01 06:00":
                close_file_handlers(file_handlers)
                file_handlers = init_file_handlers("test_data")
            for station, row in stations.items():
                prepared = row.prepare_line()
                file_handlers[station].write(f"{date};{prepared};\n")
                    
    close_file_handlers(file_handlers)
    
    
process_csv_files()
