from enum import Enum
from random import choice
from typing import List, Dict

from Mlp import MLP  # Assuming MLP is a class for language processing


class GuessStatus(Enum):
    UNSET = "WHITE"
    IN_POS = "GREEN"
    OUT_OF_POS = "YELLOW"
    MISSING = "GREY"


class Guess:
    def __init__(self, index: int):
        self.character = None
        self.status = GuessStatus.UNSET
        self.index = index

    def __str__(self):
        return f"{self.character} -> {self.status}"


class WordleGame:
    def __init__(self, wordle_words: List[str], english_model: MLP):
        self.english_model = english_model
        self.wordle_words = wordle_words

    @staticmethod
    def get_words_from_file(file_name: str) -> List[str]:
        """Reads words from a file and returns them as a list."""
        with open(file_name, "r") as file:
            words = []
            for line in file:
                words.extend(line.strip().split(','))
            return words

    @staticmethod
    def get_starter_word() -> str:
        """Returns a random starter word from a predefined list."""
        starter_words = ["oater", "realo", "alert", "irate", "ariel", "aeros", "aster", "earls", "arise", "antre"]
        return choice(starter_words)

    def _is_correct_guess(self, guesses: List[Guess]) -> bool:
        """Checks if all guesses are in the correct position."""
        return all(guess.status == GuessStatus.IN_POS for guess in guesses)

    def _match_guess(self, guessed_word: str, actual_word: str) -> List[Guess]:
        """Compares a guessed word with the actual word and provides feedback."""
        if len(guessed_word) != len(actual_word):
            print(f"Guessed Word: {guessed_word}, Actual Word: {actual_word}")
            return []

        result = []
        actual_chars = set(actual_word)
        for i, (guessed_char, actual_char) in enumerate(zip(guessed_word, actual_word)):
            guess = Guess(i)
            guess.character = guessed_char
            if guessed_char == actual_char:
                guess.status = GuessStatus.IN_POS
                actual_chars.discard(actual_char)
            else:
                guess.status = GuessStatus.UNSET
            result.append(guess)

        for i, guessed_char in enumerate(guessed_word):
            if result[i].status == GuessStatus.UNSET:
                if guessed_char in actual_chars:
                    result[i].status = GuessStatus.OUT_OF_POS
                else:
                    result[i].status = GuessStatus.MISSING

        return result

    def _make_guess(self, actual_word: str) -> List[List[Guess]]:
        """Logic for guessing the word."""
        guesses = []
        guess_word = WordleGame.get_starter_word()
        guessed_words = [guess_word]
        guess_objects = self._match_guess(guess_word, actual_word)

        guesses.append(guess_objects)
        guessed_words.append(guess_word)

        in_pos_chars, out_pos_chars, excluded_chars = {}, {}, set()

        while not self._is_correct_guess(guess_objects):
            for i, guess in enumerate(guess_objects):
                if guess.status == GuessStatus.IN_POS:
                    in_pos_chars[i] = guess.character
                    excluded_chars.discard(guess.character)
                    for key, values in out_pos_chars.items():
                        if guess.character in values:
                            values.remove(guess.character)
                elif guess.status == GuessStatus.OUT_OF_POS:
                    if i not in out_pos_chars:
                        out_pos_chars[i] = set()
                    out_pos_chars[i].add(guess.character)
                elif guess.status == GuessStatus.MISSING:
                    excluded_chars.add(guess.character)

            guess_word = english_model.sample_length_with_constraints(length=5, in_position_chars=in_pos_chars,
                                                                      out_position_chars=out_pos_chars,
                                                                      excluded_chars=excluded_chars,
                                                                      previous_samples=guessed_words)
            guesses.append(guess_objects)
            guessed_words.append(guess_word)
            guess_objects = self._match_guess(guess_word, actual_word)

        return guesses

    def play_game(self):
        """Main function to initiate the game."""

        guesses_map: Dict[str, List[List[Guess]]] = {}

        for word in self.wordle_words:
            list_of_guesses = self._make_guess(word)
            guesses_map[word] = list_of_guesses

        total_guesses = sum(len(val) for val in guesses_map.values())
        for word, guesses in guesses_map.items():
            print(f"Word: {word} -> Guesses: {len(guesses)}")

        for word, guesses in guesses_map.items():
            print(f"Word: {word} ->")
            for guess in guesses:
                print("   " + ", ".join(str(g) for g in guess))

        total_wordles = len(self.wordle_words)
        print(f"Total Chances: {total_guesses}")
        print(f"Total Wordles in game: {total_wordles}")

        average_guesses = total_guesses / total_wordles if total_wordles != 0 else 1
        print(f"Average Chances: {average_guesses}")


if __name__ == "__main__":
    english_model = MLP(vocab_size=27, context_size=4, embedding_dim=10, tanh_dim=200)
    english_model.read_words("english.txt")
    english_model.build_datasets()
    english_model.train(batch_size=32, num_epochs=10)
    wordle_game = WordleGame(WordleGame.get_words_from_file("wordle.txt"), english_model)
    wordle_game.play_game()
