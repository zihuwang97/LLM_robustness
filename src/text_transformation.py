import numpy as np
import torch
import random
import string
import nltk
nltk.download('punkt')


class TextTransformation():
    def __init__(self):
        super().__init__()

    def apply(self, text, perturbation):
        text_tokenized = nltk.word_tokenize(text)
        random_word_index = 0
        random_word_selected = False
        for _ in range(10):
            if random_word_selected == True:
                break
            random_word_index = self.return_random_number(0, len(text_tokenized)-1)
            if (len(text_tokenized[random_word_index]) > 2):
                random_word_selected = True
        if random_word_selected == True:
            selected_word = text_tokenized[random_word_index]
            if perturbation == "delete-char":
                perturbed_word = self.deletion_char(selected_word)
            elif perturbation == "insertion-char":
                perturbed_word = self.insertion_char(selected_word)
            elif perturbation == "letter-case-change-char":
                perturbed_word = self.letter_case_change_char(selected_word)
            elif perturbation == "repetition-char":
                perturbed_word = self.repetition_char(selected_word)
            elif perturbation == "replacement-char":
                perturbed_word = self.replacement_char(selected_word)
            elif perturbation == "swap-char":
                perturbed_word = self.swap_char(selected_word)
            elif perturbation == "delete-word":
                perturbed_word = ""
            elif perturbation == "repetition-word":
                perturbed_word = selected_word + ' ' + selected_word + ' '
            else:
                perturbed_word = selected_word
        else:
            return text

        perturbed_sample = ""
        for i in range(0, random_word_index):
            perturbed_sample += text_tokenized[i] + ' '
        perturbed_sample += perturbed_word + ' '
        for i in range(random_word_index+1, len(text_tokenized)): 
            if i == (len(text_tokenized) - 1):
                perturbed_sample += text_tokenized[i]
            else:
                perturbed_sample += text_tokenized[i] + ' '
        return perturbed_sample

    def return_random_number(self, begin, end):
        return random.randint(begin, end)

    def random_changing_type(self):
        random_num = random.randint(1, 2)
        if (random_num == 1):
            return 'FirstChar'
        else:
            return 'AllChars'
        
    def swap_characters(self, input_word, position, adjacent):
        temp_word = ''
        if (adjacent == 'left'):
            if (position == 1):
                temp_word = input_word[1]
                temp_word += input_word[0]
                temp_word += input_word[2:]
            elif (position == len(input_word)-1):
                temp_word = input_word[0:position-1]
                temp_word += input_word[position]
                temp_word += input_word[position-1]
            elif (position > 1 and position < len(input_word)-1):
                temp_word = input_word[0:position-1]
                temp_word += input_word[position]
                temp_word += input_word[position-1]
                temp_word += input_word[position+1:]
                
        elif (adjacent == 'right'):
            if (position == 0):
                temp_word = input_word[1]
                temp_word += input_word[0]
                temp_word += input_word[2:]
            elif (position == len(input_word)-2):
                temp_word = input_word[0:position]
                temp_word += input_word[position+1]
                temp_word += input_word[position]
            elif (position > 0 and position < len(input_word)-2):
                temp_word = input_word[0:position]
                temp_word += input_word[position+1]
                temp_word += input_word[position]
                temp_word += input_word[position+2:]
                
        return temp_word
    
    def change_ordering(self, input_length, input_side, input_changes):
        ordering = []
        if (input_side == 1):
            for i in range(0, input_length):
                if (i < input_changes):
                    candidates=[]
                    for j in range(0, input_changes):
                        if (j != i and j not in ordering):
                            candidates.append(j)
                            
                    if (len(candidates) > 0):
                        random_index = self.return_random_number(0, len(candidates)-1)
                        ordering.append(candidates[random_index])
                    else:
                        ordering.append(i)
                else:
                    ordering.append(i)
        elif (input_side == 2):
            for i in range(0, input_length):
                if (i < input_length-input_changes):
                    ordering.append(i)
                else:
                    candidates=[]
                    for j in range(input_length-input_changes, input_length):
                        if (j != i and j not in ordering):
                            candidates.append(j)
                    if (len(candidates) > 0):
                        random_index = self.return_random_number(0, len(candidates)-1)
                        ordering.append(candidates[random_index])
                    else:
                        ordering.append(i)
        return 

    def deletion_char(self, selected_word):
        random_char_index = self.return_random_number(1, len(selected_word)-2)
        perturbed_word = selected_word[:random_char_index]
        perturbed_word += selected_word[random_char_index+1:]
        return perturbed_word

    def insertion_char(self, selected_word):
        random_char_index = self.return_random_number(1, len(selected_word)-2)
        random_char_code = self.return_random_number(97, 122)
        perturbed_word = selected_word[:random_char_index]
        perturbed_word += chr(random_char_code)
        perturbed_word += selected_word[random_char_index:]
        return perturbed_word

    def letter_case_change_char(self, selected_word):
        perturbed_word = ""
        change_type = self.random_changing_type()
        if (change_type == 'FirstChar'):
            print('Letter case changing: First character')
            if (ord(selected_word[0]) >= 97 and ord(selected_word[0]) <= 122):
                perturbed_word = chr(ord(selected_word[0])-32)
                perturbed_word += selected_word[1:]
            elif (ord(selected_word[0]) >= 65 and ord(selected_word[0]) <= 90):
                perturbed_word = chr(ord(selected_word[0])+32)
                perturbed_word += selected_word[1:]
            else:
                perturbed_word = selected_word

        elif (change_type == 'AllChars'):
            print('Letter case changing: All characters')
            for i in range(0, len(selected_word)):
                if (ord(selected_word[i]) >= 97 and ord(selected_word[i]) <= 122):
                    perturbed_word += chr(ord(selected_word[i])-32)
                elif (ord(selected_word[i]) >= 65 and ord(selected_word[i]) <= 90):
                    perturbed_word += chr(ord(selected_word[i])+32)
                else:
                    perturbed_word += selected_word[i]
        return perturbed_word

    def repetition_char(self, selected_word):
        random_char_index = self.return_random_number(1, len(selected_word)-2)
        perturbed_word = selected_word[:random_char_index]
        perturbed_word += selected_word[random_char_index] + selected_word[random_char_index]
        perturbed_word += selected_word[random_char_index+1:]
        return perturbed_word

    def replacement_char(self, selected_word):
        char_is_letter = False
        tries_number = 0
        while (char_is_letter != True and tries_number <= 20):
            random_char_index = self.return_random_number(1, len(selected_word)-2)
            tries_number += 1
            if ((ord(selected_word[random_char_index]) >= 97 and ord(selected_word[random_char_index]) <= 122) or (ord(selected_word[random_char_index]) >= 65 and ord(selected_word[random_char_index]) <= 90)):
                char_is_letter = True
        char_to_replace = selected_word[random_char_index]
        adjacent_char = random.choice(string.ascii_letters)
        perturbed_word = selected_word[:random_char_index]
        perturbed_word += adjacent_char
        perturbed_word += selected_word[random_char_index+1:]
        return perturbed_word

    def swap_char(self, selected_word):
        random_char_index = self.return_random_number(0, len(selected_word)-1)
        adjacent_for_swapping = ''
        if (random_char_index == 0):
            adjacent_for_swapping = 'right'
        elif (random_char_index == len(selected_word)-1):
            adjacent_for_swapping = 'left'
        else:
            adjacent = self.return_random_number(1, 2)
            if(adjacent == 1):
                adjacent_for_swapping = 'left'
            else:
                adjacent_for_swapping = 'right'
        perturbed_word = self.swap_characters(selected_word, random_char_index, adjacent_for_swapping)
        return perturbed_word