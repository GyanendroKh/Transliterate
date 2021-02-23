class Lang:
    en = 'en'
    mm = 'mm'
    bn = 'bn'
    __langs = None

    @staticmethod
    def get_langs():
        if not Lang.__langs:
            Lang.__langs = [
                f for f in vars(Lang)
                if not f.startswith('_') and not callable(getattr(Lang, f))
            ]
        return Lang.__langs

    @staticmethod
    def is_supported(lang):
        return lang in Lang.get_langs()


class Tokens:
    start = '<start>'
    end = '<end>'
    pad = '<pad>'
    unk = '<unk>'

    @staticmethod
    def to(lang):
        return f'<2{lang}>'


known_ext = ['csv', 'tab']
