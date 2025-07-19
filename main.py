"""Data preparation"""
# Load data from a text file
text = ""
with open('polish_text.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Remove all punctuation marks
import string
text = text.translate(str.maketrans('', '', string.punctuation + "—" + "…" + "“" + "”" + "‘" + "’" +"„"))

# Convert all letters to lowercase
text = text.lower()

# Split text into words
words = text.split()

# Create a dictionary of unique words
unique_words = set(words)

# Create a dictionary with the count of each word's occurrences
word_count = {}
for word in words:
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 1

# Create a dictionary with probabilities of each word's occurrences
total_words = len(words)
word_probabilities = {}
for word, count in word_count.items(): # (word, count) pairs from word_count
    word_probabilities[word] = count / total_words

"""In summary we have:
- Total words: total_words
- Dictionary of unique words: unique_words
- Dictionary with count of each word's occurrences: word_count (word, count)
- Dictionary with probabilities of each word's occurrences: word_probabilities (word, probability)
- Words grouped by length: words_by_length (length, [words])
"""



"""Main loop"""
import curses
import numpy as np

def levenshtein_distance(s1, s2):
    # s1 = source, s2 = target
    n = len(s1) + 1  # add an empty character at the beginning of s1
    m = len(s2) + 1  # add an empty character at the beginning of s2
    matrix = np.zeros((n, m), dtype=int)

    # The two loops below are used so that we don't have to check bounds in the main loop.
    # The number of operations needed to convert s1[:i] to an empty string (i.e., deletions)
    for i in range(n):
        matrix[i][0] = i

    # The number of operations needed to convert an empty string to s2[:j] (i.e., insertions)
    for j in range(m):
        matrix[0][j] = j

    # Compute the Levenshtein distance
    # In each iteration we consider 4 values
    for i in range(1, n):
        for j in range(1, m):
            if s1[i - 1] == s2[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]  # Characters are the same, no operation needed
            else:
                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,     # Deletion
                    matrix[i][j - 1] + 1,     # Insertion
                    matrix[i - 1][j - 1] + 1  # Substitution
                )

    return matrix[n - 1][m - 1]

def autocorrect(word, edit_distance=3):
    # Preprocess the input word
    word = word.strip().lower()
    word = word.translate(str.maketrans('', '', string.punctuation + "—" + "…" + "“" + "”" + "‘" + "’" +"„")) # Remove punctuation and special characters

    # Check if there is a mistake in the word
    if word in unique_words:
        return word # Word is correct, do nothing
    
    # else: correct the word

    # Find words with the same length or similar length
    suggested_words_length = [len(word)] # Can be adjusted to include more lengths, e.g., [len(word) - 1, len(word), len(word) + 1]
    
    # Find similar words
    similar_words = []
    for length in suggested_words_length:
        if length in words_by_length:
            for candidate in words_by_length[length]:
                # Check if the candidate word is similar enough
                if levenshtein_distance(word, candidate) <= edit_distance and candidate.startswith(word[0]):
                    similar_words.append(candidate)

    # If no similar words found, return the original word (it might be correct but not in the dictionary)
    if not similar_words:
        return word
    
    # Otherwise, select the most probable correction among the closest words
    
    # Sort by Levenshtein distance (Smallest first)
    similar_words.sort(key=lambda x: levenshtein_distance(word, x), reverse=False)

    # Get all words with minimum Levenshtein distance
    min_distance = levenshtein_distance(word, similar_words[0])
    temp = [w for w in similar_words if levenshtein_distance(word, w) == min_distance]

    # Sort by probability (Highest first)
    temp.sort(key=lambda x: word_probabilities.get(x, 0), reverse=True)

    # Return the most probable word
    return temp[0]

def backspace(stdscr):
    y, x = stdscr.getyx()
    if x > 0:
        stdscr.move(y, x - 1)
        stdscr.delch()
    elif x == 0 and y > 0:
        stdscr.move(y - 1, curses.COLS - 1)
        stdscr.delch()


if __name__ == "__main__":
    # Initialize curses
    stdscr = curses.initscr()
    curses.noecho()
    stdscr.clear()

    input_buffer = []
    
    while True:
        char = stdscr.get_wch()
        input_buffer.append(char)

        # Enter key
        if char == '\n':
            stdscr.clear()
            input_buffer = []

        elif char == '\x7f':  # Backspace key
            backspace(stdscr)
            if input_buffer[-1] == '\x7f':  # If the last character is a backspace
                input_buffer.pop() # Remove backspace character from the input buffer
            if input_buffer:
                input_buffer.pop()

        elif char == ' ':
            words = ''.join(input_buffer).split()

            last_word = words[-1] if words else ''
            most_probable_word = autocorrect(last_word)
            if most_probable_word:
                # Remove the last word from the input buffer
                input_buffer = input_buffer[:-len(last_word) - 1]
                # Add the most probable word to the input buffer
                input_buffer.extend(most_probable_word + ' ')

        elif char == '\x1b':  # Escape key
            curses.endwin()
            break

        stdscr.clear()
        stdscr.addstr("".join(input_buffer))
        stdscr.refresh()