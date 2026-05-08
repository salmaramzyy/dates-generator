from typing import Dict, List, Tuple


DAY_TOKENS: Dict[str, int] = {
    "MON": 0, "TUE": 1, "WED": 2, "THU": 3,
    "FRI": 4, "SAT": 5, "SUN": 6
}

MONTH_TOKENS: Dict[str, int] = {
    "JAN": 0, "FEB": 1, "MAR": 2, "APR": 3,
    "MAY": 4, "JUN": 5, "JUL": 6, "AUG": 7,
    "SEP": 8, "OCT": 9, "NOV": 10, "DEC": 11
}

LEAP_TOKENS: Dict[str, int] = {"False": 0, "True": 1}

MIN_DECADE = 180
MAX_DECADE = 220


CHAR_TO_TOKEN: Dict[str, int] = {str(i): i for i in range(10)}
CHAR_TO_TOKEN["-"] = 10
CHAR_TO_TOKEN["PAD"] = 11
CHAR_TO_TOKEN["SOS"] = 12
CHAR_TO_TOKEN["EOS"] = 13

TOKEN_TO_CHAR: Dict[int, str] = {v: k for k, v in CHAR_TO_TOKEN.items()}

PAD_TOKEN = 11
SOS_TOKEN = 12
EOS_TOKEN = 13
VOCAB_SIZE = 14           
DATE_SEQ_LEN = 10        


class DateTokenizer:
    """Converts raw data lines to tensors and back."""


    def encode_conditions(self, day: str, month: str,
                          leap: str, decade: str) -> Tuple[int, int, int, int]:
        """
        Returns (day_idx, month_idx, leap_idx, decade_idx).
        Example: encode_conditions('WED','JAN','False','180') → (2,0,0,0)
        """
        day_idx    = DAY_TOKENS[day]
        month_idx  = MONTH_TOKENS[month]
        leap_idx   = LEAP_TOKENS[leap]
        decade_idx = int(decade) - MIN_DECADE
        return day_idx, month_idx, leap_idx, decade_idx


    def normalize_date(self, date_str: str) -> str:
        """
        '1-1-1800' → '01-01-1800'
        Always returns exactly 10 characters.
        """
        day, month, year = date_str.strip().split("-")
        return f"{int(day):02d}-{int(month):02d}-{year}"

    def encode_date(self, date_str: str) -> List[int]:
        """
        '1-1-1800' → [0, 1, 10, 0, 1, 10, 1, 8, 0, 0]
        """
        normalized = self.normalize_date(date_str)
        return [CHAR_TO_TOKEN[ch] for ch in normalized]

    def decode_date(self, tokens: List[int]) -> str:
        """
        [0, 1, 10, 0, 1, 10, 1, 8, 0, 0] → '01-01-1800'
        Stops at EOS or PAD.
        """
        chars = []
        for t in tokens:
            if t in (EOS_TOKEN, PAD_TOKEN, SOS_TOKEN):
                break
            chars.append(TOKEN_TO_CHAR[t])
        return "".join(chars)


    def parse_line(self, line: str) -> Tuple[Tuple[int,int,int,int], List[int]]:
        """
        Parses one line from data.txt.
        Returns (conditions_tuple, date_tokens).
        Example:
          '[WED] [JAN] [False] [180] 1-1-1800'
          → ((2,0,0,0), [0,1,10,0,1,10,1,8,0,0])
        """
        parts = line.strip().split()
        day    = parts[0][1:-1]   # remove [ and ]
        month  = parts[1][1:-1]
        leap   = parts[2][1:-1]
        decade = parts[3][1:-1]
        date   = parts[4]

        conditions = self.encode_conditions(day, month, leap, decade)
        date_tokens = self.encode_date(date)
        return conditions, date_tokens

    def parse_input_line(self, line: str) -> Tuple[int, int, int, int]:
        """
        Parses a line with NO date (for inference).
        Example: '[WED] [JAN] [False] [180]' → (2,0,0,0)
        """
        parts = line.strip().split()
        day    = parts[0][1:-1]
        month  = parts[1][1:-1]
        leap   = parts[2][1:-1]
        decade = parts[3][1:-1]
        return self.encode_conditions(day, month, leap, decade)


if __name__ == "__main__":
    tok = DateTokenizer()

    line = "[WED] [JAN] [False] [180] 1-1-1800"
    conditions, date_tokens = tok.parse_line(line)

    print("Conditions:", conditions)
    print("Date tokens:", date_tokens)
    print("Decoded date:", tok.decode_date(date_tokens))
    print("Normalized:", tok.normalize_date("1-1-1800"))