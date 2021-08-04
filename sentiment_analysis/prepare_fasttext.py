"Prepare fastText for loading with lower RAM footprint."


import logging
import pathlib

import gensim


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    folder = pathlib.Path(__file__).parent
    gft_model = gensim.models.fasttext.load_facebook_vectors(
        str(folder / 'cc.de.300.bin')
    )
    gft_model.save(str(folder / 'cc.de.300.bin.gensim_model'))

