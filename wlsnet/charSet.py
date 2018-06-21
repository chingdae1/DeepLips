en_list = (
    'A',
    'B',
    'C',
    'D',
    'E',
    'F',
    'G',
    'H',
    'I',
    'J',
    'K',
    'L',
    'M',
    'N',
    'O',
    'P',
    'Q',
    'R',
    'S',
    'T',
    'U',
    'V',
    'W',
    'X',
    'Y',
    'Z',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '0',
    '<sos>',
    '<eos>',
    '<pad>',
    '\'',
)

kr_list = (
    'ㄱ',
    'ㄴ',
    'ㄷ',
    'ㄹ',
    'ㅁ',
    'ㅂ',
    'ㅅ',
    'ㅇ',
    'ㅈ',
    'ㅊ',
    'ㅋ',
    'ㅌ',
    'ㅍ',
    'ㅎ',
    'ㅛ',
    'ㅕ',
    'ㅑ',
    'ㅐ',
    'ㅔ',
    'ㅗ',
    'ㅓ',
    'ㅏ',
    'ㅣ',
    'ㅠ',
    'ㅜ',
    'ㅡ',
    'ㅃ',
    'ㅉ',
    'ㄸ',
    'ㄲ',
    'ㅆ',
    'ㅒ',
    'ㅖ',
    'ㄻ',
    'ㄶ',
    'ㄼ',
    'ㅀ',
    'ㄺ',
    'ㅘ',
    'ㅙ',
    'ㅝ',
    'ㅟ',
    'ㅚ',
    'ㅞ',
    'ㅢ',
    'ㅄ',
    'ㄵ',
    '<sos>',
    '<eos>',
    '<pad>'
)

class CharSet:
    def __init__(self, language):
        if language == 'kr':
            self.__char_list = kr_list
        else:
            self.__char_list = en_list
        _init_index_dict()
    
    def _init_index_dict(self):
        index_list = [i for i in range(len(self.char_list))]
        self.__index_of_str = dict(zip(self.char_list, index_list))
        self.__char_of_index = dict(zip(index_list, self.char_list))
        self.__total_num = len(index_list)

    def get_index_of(self, string):
        return self.__index_of_str[string]

    def get_char_of(self, index):
        return self.__char_of_index[index]

    def get_total_num(self):
        return self.__total_num