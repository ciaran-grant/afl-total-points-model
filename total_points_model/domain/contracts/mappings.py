from dataclasses import dataclass

@dataclass
class Mappings:
    mappings = {
        'Round': {
            '01':1,
            '02':2,
            '03':3,
            '04':4,
            '05':5,
            '06':6,
            '07':7,
            '08':8,
            '09':9,
            '10':10,
            '11':11,
            '12':12,
            '13':13,
            '14':14,
            '15':15,
            '16':16,
            '17':17,
            '18':18,
            '19':19,
            '20':20,
            '21':21,
            '22':22,
            '23':23,
            '24':24,
            'F1':25,
            'F2':26,
            'F3':27,
            'F4':28,
            'F5':29
        },
        "Weather_Type" : {
            "CLEAR_NIGHT":"Good",
            "MOSTLY_CLEAR":"Good",
            "MOSTLY_SUNNY":"Good",
            "OVERCAST":"Good",
            "RAIN":"Bad",
            "ROOF_CLOSED":"Good",
            "SUNNY":"Good",
            "THUNDERSTORMS":"Bad",
            "WINDY":"Bad",
            "WINDY_RAIN":"Bad"
        }
    }
    