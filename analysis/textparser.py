import pandas as pd
import numpy as np
import nltk

class TextParser():
    src_imported: bool = False       
    src_clipped: bool = False
    src_col_suffix: str = '_str'

    join_pat: str = r'\n'
    strip_hyphens: bool = False
    strip_whitespace: bool = False
    verbose: bool = False

    ohco_pats: [] = [
        ('para', r"\n\n", 'd'),
        ('sent', r"[.?!;:]+", 'd'),
        ('token', r"[\s',-]+", 'd')
    ]

    _ohco_type: {} = {
        'd': '_num',
        'm': '_id'
    }

    def __init__(self, src_file: str, ohco_pats: [], clip_pats: [], use_nltk=True):
        self.src_file = src_file            
        self.clip_pats = clip_pats
        self.ohco_pats = ohco_pats + self.ohco_pats
        self.OHCO = [item[0]+self._ohco_type[item[2]] for item in self.ohco_pats]
        self.ohco_names = [item[0] for item in self.ohco_pats]
        self.use_nltk = use_nltk

        if self.use_nltk:
            self.ohco_pats[-2] = ('sent', None, 'nltk')
            self.ohco_pats[-1] = ('token', None, 'nltk')
            for package in [
                'tokenizers/punkt', 
                'taggers/averaged_perceptron_tagger', 
                'corpora/stopwords', 
                'help/tagsets'
            ]:
                try:
                    nltk.data.find(package)
                except IndexError:
                    nltk.download(package)

    def import_source(self, strip: bool = True, char_encoding: str = "utf-8-sig"):
        if self.verbose: print("Importing ", self.src_file)
        text_lines = open(self.src_file, 'r', encoding=char_encoding).readlines()
        self.LINES = pd.DataFrame({'line_str': text_lines})
        self.LINES.index.name = 'line_id'
        if strip:
            self.LINES.line_str = self.LINES.line_str.str.strip()
        self.src_imported = True
        if self.verbose: print("Clipping text")
        self._clip_lines()
        return self        

    def _clip_lines(self):
        start_pat = self.clip_pats[0]
        end_pat = self.clip_pats[1]
        start = self.LINES.line_str.str.contains(start_pat, regex=True)
        end = self.LINES.line_str.str.contains(end_pat, regex=True)
        try:
            start_line_num = self.LINES.loc[start].index[0]
        except IndexError:
            raise ValueError("Clip start pattern not found.")            
        try:
            end_line_num = self.LINES.loc[end].index[0]
        except IndexError:
            raise ValueError("Clip end pattern not found.")
        self.LINES = self.LINES.loc[start_line_num + 1: end_line_num - 2]
        self.src_clipped = True

    def parse_tokens(self):
        """Convert lines to tokens based on OHCO."""
        if not self.src_imported:
            raise RuntimeError("Source not imported. Please run .import_source()")
    
        self.TOKENS = self.LINES.copy()
    
        for i, level in enumerate(self.OHCO):
            if self.verbose:
                print(f"Parsing OHCO level {i} {level}", end=' ')
    
            parse_type = self.ohco_pats[i][2]
            div_name = self.ohco_pats[i][0]
            div_pat = self.ohco_pats[i][1]
    
            src_div_name = 'line' if i == 0 else self.ohco_names[i - 1]
            src_col = f"{src_div_name}{self.src_col_suffix}"
            dst_col = f"{div_name}{self.src_col_suffix}"
    
            if parse_type == 'm':
                if self.verbose: print(f"by milestone {div_pat}")
                div_lines = self.TOKENS[src_col].str.contains(div_pat, regex=True, case=True)
                self.TOKENS.loc[div_lines, div_name] = [j + 1 for j in range(div_lines.sum())]
                self.TOKENS[div_name] = self.TOKENS[div_name].ffill()
                self.TOKENS = self.TOKENS.loc[~self.TOKENS[div_name].isna()]
                self.TOKENS = self.TOKENS.loc[~div_lines]
                self.TOKENS[div_name] = self.TOKENS[div_name].astype('int')
                self.TOKENS = self.TOKENS.groupby(self.ohco_names[:i + 1], group_keys=True)[src_col]\
                    .apply(lambda x: '\n'.join(x)).to_frame(dst_col)
    
            elif parse_type == 'd':
                if self.verbose: print(f"by delimitter {div_pat}")
                self.TOKENS = self.TOKENS[src_col].str.split(div_pat, expand=True).stack().to_frame(dst_col)
    
            elif parse_type == 'nltk' and level == 'sent_num':
                if self.verbose: print("by NLTK sentence tokenizer")
                
                # Force DataFrame structure for expected column
                if isinstance(self.TOKENS, pd.Series):
                    self.TOKENS = self.TOKENS.to_frame(name='para_str')
                elif isinstance(self.TOKENS, pd.DataFrame) and 'para_str' not in self.TOKENS.columns:
                    self.TOKENS.columns = ['para_str']
                
                # Apply sentence tokenizer and ensure result is a DataFrame
                sent_df = pd.DataFrame(self.TOKENS['para_str'].apply(lambda x: pd.Series(nltk.sent_tokenize(x))))
                self.TOKENS = sent_df.stack().to_frame('sent_str')
    
            elif parse_type == 'nltk' and level == 'token_num':
                if self.verbose: print("by NLTK tokenization")
                if self.strip_hyphens:
                    if 'sent_str' in self.TOKENS.columns:
                        self.TOKENS['sent_str'] = self.TOKENS['sent_str'].astype(str).str.replace(r"-", " ", regex=True)

                if self.strip_whitespace:
                    tokenizer = nltk.WhitespaceTokenizer()
                else:
                    tokenizer = nltk.word_tokenize
    
                # Apply POS tagging and ensure the result is a DataFrame
                self.TOKENS = self.TOKENS['sent_str'].apply(
                    lambda x: pd.Series(nltk.pos_tag(tokenizer.tokenize(str(x))), dtype='object')
                )
                
                # Now stack it
                if isinstance(self.TOKENS, pd.Series):
                    self.TOKENS = self.TOKENS.to_frame()
                self.TOKENS = self.TOKENS.stack().to_frame('pos_tuple')
                
                # Unpack the tagged output
                self.TOKENS['pos'] = self.TOKENS['pos_tuple'].apply(lambda x: x[1])
                self.TOKENS['token_str'] = self.TOKENS['pos_tuple'].apply(lambda x: x[0])
                self.TOKENS['term_str'] = self.TOKENS['token_str'].str.lower()
    
            else:
                raise ValueError(f"Invalid parse option: {parse_type}.")
    
            self.TOKENS.index.names = self.OHCO[:i + 1]
    
        if self.use_nltk:
            punc_pos = ['$', "''", '(', ')', ',', '--', '.', ':', '``']
            self.TOKENS['term_str'] = self.TOKENS[~self.TOKENS.pos.isin(punc_pos)].token_str\
                .str.replace(r'[\W_]+', '', regex=True).str.lower()
        else:
            self.TOKENS['term_str'] = self.TOKENS.token_str.str.replace(r'[\W_]+', '', regex=True).str.lower()


    def extract_vocab(self):
        self.VOCAB = self.TOKENS.term_str.value_counts().to_frame('n')
        self.VOCAB.index.name = 'term_str'
        self.VOCAB['n_chars'] = self.VOCAB.index.str.len()
        self.VOCAB['p'] = self.VOCAB['n'] / self.VOCAB['n'].sum()
        self.VOCAB['s'] = 1 / self.VOCAB['p']
        self.VOCAB['i'] = np.log2(self.VOCAB['s'])
        self.VOCAB['h'] = self.VOCAB['p'] * self.VOCAB['i']
        self.H = self.VOCAB['h'].sum()
        return self

    def gather_tokens(self, level=0, grouping_col='term_str', cat_sep=' '):
        max_level = len(self.OHCO) - 2
        if level > max_level:
            raise ValueError(f"Level {level} too high. Try between 0 and {max_level}")
        level_name = self.OHCO[level].split('_')[0]
        idx = self.TOKENS.index.names[:level+1]
        return self.TOKENS.groupby(idx)[grouping_col]\
            .apply(lambda x: x.str.cat(sep=cat_sep)).to_frame(f'{level_name}_str')
