import pyonmttok
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-file",
        required=True,
        help="input file to tokenize",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="output file destination",
    )
    parser.add_argument(
        "--bpe-model",
        required=True,
        help="pretrained bpe model",
    )

    args = parser.parse_args()

    tokenizer = pyonmttok.Tokenizer(
        mode="aggressive",
        lang="fr",
        joiner_annotate=True,
        bpe_model_path=args.bpe_model,
    )
    tokenizer.tokenize_file(
        args.input_file,
        args.output_file,
    )
