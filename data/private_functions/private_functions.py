PRIVATE_FUNCTIONS = [
    { 
        "id": "private_1", 
        "source": "private", 
        "task_description": "Reverse words whose length is odd.", 
        "function_name": "reverse_odd_words", 
        "code": """ 
            def reverse_odd_words(words): 
                return [w[::-1] if len(w) % 2 else w for w in words] 
            """ 
    }, 
    { 
        "id": "private_2", 
        "source": "private", 
        "task_description": "Add even-indexed numbers and subtract odd-indexed ones.", 
        "function_name": "alternating_sum", 
        "code": """ 
            def alternating_sum(nums): 
                return sum(nums[::2]) - sum(nums[1::2]) 
            """ 
    }, 
    { 
        "id": "private_3", 
        "source": "private", 
        "task_description": "Wrap each word in the text with a given marker.", 
        "function_name": "wrap_text", 
        "code": """ 
            def wrap_text(text, marker): 
                return " ".join(marker + w + marker for w in text.split()) 
            """ 
    }, 
    { 
        "id": "private_4", 
        "source": "private", 
        "task_description": "Count how many times the sign of numbers changes in a list.", 
        "function_name": "count_switch_points", 
        "code": """ 
            def count_switch_points(values): 
                count = 0 
                for i in range(1, len(values)):
                    if values[i] * values[i - 1] < 0: 
                        count += 1 
                return count 
            """ 
    }, 
    { 
        "id": "private_5", 
        "source": "private", 
        "task_description": "Duplicate the last n elements of a list.", 
        "function_name": "duplicate_last_n", 
        "code": """ 
            def duplicate_last_n(lst, n): 
                return lst + lst[-n:] 
            """ 
    }, 
    { 
        "id": "private_6", 
        "source": "private", 
        "task_description": "Merge two lists by selecting the element with shorter string length at each index.", 
        "function_name": "merge_by_length", 
        "code": """ 
            def merge_by_length(a, b): 
                return [ 
                    a[i] if len(str(a[i])) < len(str(b[i])) else b[i] 
                    for i in range(len(a)) 
                ] 
            """ 
    }, 
    { 
        "id": "private_7", 
        "source": "private", 
        "task_description": "Shift each vowel in the text to the next vowel in sequence.", 
        "function_name": "shift_vowels", 
        "code": """ 
            def shift_vowels(text): 
                vowels = "aeiou" 
                result = "" 
                for ch in text: 
                    if ch in vowels: 
                        result += vowels[(vowels.index(ch) + 1) % 5] 
                    else: 
                        result += ch 
                return result 
            """ 
    }, 
    { 
        "id": "private_8", 
        "source": "private", 
        "task_description": "Square numbers greater than a threshold and keep others unchanged.", 
        "function_name": "conditional_multiply", 
        "code": """ 
            def conditional_multiply(nums, threshold): 
                return [x * x if x > threshold else x for x in nums] 
            """ 
    }, 
    { 
        "id": "private_9", 
        "source": "private", 
        "task_description": "Flatten all list values of a dictionary into a single list.", 
        "function_name": "flatten_dict_values", 
        "code": """ 
            def flatten_dict_values(d): 
                out = [] 
                for v in d.values(): 
                    out.extend(v) 
                return out 
            """ 
    }, 
    { 
        "id": "private_10", 
        "source": "private", 
        "task_description": "Return the longest prefix of the list where the cumulative sum never decreases.", 
        "function_name": "non_decreasing_prefix_sum", 
        "code": """ 
            def non_decreasing_prefix_sum(nums): 
                if not nums: 
                    return [] 
                result = [nums[0]] 
                current_sum = nums[0] 
                for x in nums[1:]: 
                    if current_sum + x < current_sum: 
                        break 
                    current_sum += x 
                    result.append(x) 
                return result 
            """ 
    }, 
    { 
        "id": "private_11", 
        "source": "private", 
        "task_description": "Return characters located at even indices in a string.", 
        "function_name": "extract_even_positions", 
        "code": """ 
            def extract_even_positions(text): 
                return "".join(text[i] for i in range(0, len(text), 2)) 
            """ 
    }, 
    { 
        "id": "private_12", 
        "source": "private", 
        "task_description": "Rotate a list to the right by k positions.", 
        "function_name": "rotate_right", 
        "code": """ 
            def rotate_right(lst, k): 
                k %= len(lst) 
                return lst[-k:] + lst[:-k] 
            """ 
    }, 
    { 
        "id": "private_13", 
        "source": "private", 
        "task_description": "Replace words starting with a prefix by a given token.", 
        "function_name": "replace_if_prefix", 
        "code": """ 
            def replace_if_prefix(words, prefix, token): 
                return [token if w.startswith(prefix) else w for w in words] 
            """ 
    }, 
    { 
        "id": "private_14", 
        "source": "private", 
        "task_description": "Return sorted unique characters from a string.", 
        "function_name": "unique_sorted_chars", 
        "code": """ 
            def unique_sorted_chars(text): 
                return "".join(sorted(set(text))) 
            """ 
    }, 
    { 
        "id": "private_15", 
        "source": "private", 
        "task_description": "Map each key through a dictionary with a fallback default value.", 
        "function_name": "map_with_default", 
        "code": """ 
            def map_with_default(keys, mapping, default): 
                return [mapping.get(k, default) for k in keys] 
            """ 
    }, 
    { 
        "id": "private_16", 
        "source": "private", 
        "task_description": "Summarize consecutive repeated values as (value, count).", 
        "function_name": "summarize_runs", 
        "code": """ 
            def summarize_runs(values): 
                if not values: 
                    return [] 
                out = [] 
                current = values[0] 
                count = 1 
                for v in values[1:]: 
                    if v == current: 
                        count += 1 
                    else: 
                        out.append((current, count)) 
                        current = v 
                        count = 1 
                out.append((current, count)) 
                return out 
            """ 
    }, 
    { 
        "id": "private_17", 
        "source": "private", 
        "task_description": "Filter numbers whose digit sum equals a target value.", 
        "function_name": "filter_by_digit_sum", 
        "code": """ 
            def filter_by_digit_sum(nums, target): 
                return [ 
                    n for n in nums 
                    if sum(int(d) for d in str(n)) == target 
                ] 
            """ 
    }, 
    { 
        "id": "private_18", 
        "source": "private", 
        "task_description": "Uppercase characters at even indices and lowercase at odd indices.", 
        "function_name": "toggle_case_indices", 
        "code": """ 
            def toggle_case_indices(text): 
                return "".join( 
                    ch.upper() if i % 2 == 0 else ch.lower() 
                    for i, ch in enumerate(text) 
                ) 
            """ 
    }, 
    { 
        "id": "private_19", 
        "source": "private", 
        "task_description": "Split text whenever a marker string is encountered.", 
        "function_name": "split_on_marker", 
        "code": """ 
            def split_on_marker(text, marker): 
                return text.split(marker) 
            """ 
    }, 
    { 
        "id": "private_20", 
        "source": "private", 
        "task_description": "Return the absolute difference if it is below a limit, otherwise return the limit.", 
        "function_name": "bounded_difference", 
        "code": """ 
            def bounded_difference(a, b, limit): 
                diff = abs(a - b) 
                return diff if diff < limit else limit 
            """ 
    }, 
    {
        "id": "private_21",
        "source": "private",
        "task_description": "Limit consecutive repetitions of values to k times.",
        "function_name": "limit_repeatitions",
        "code": """
            def limit_repeatitions(vals, k):
                if k <= 0:
                    return vals[:]
                result = []
                count = 0
                previous = None
                for x in vals:
                    if x == previous:
                        count += 1
                    else:
                        count = 1
                    if count <= k:
                        result.append(x)
                    previous = x
                return result
            """
    },
    {
        "id": "private_22",
        "source": "private",
        "task_description": "Build a new list by alternating elements from two lists in a zigzag pattern.",
        "function_name": "zigzag_join",
        "code": """
            def zigzag_join(list_a, list_b):
                result = []
                for x, y in zip(list_a, list_b):
                    result += [x, y]
                return result
            """
    },
    {
        "id": "private_23",
        "source": "private",
        "task_description": "Transfrom an input string so that each digit d repeats the previous character d times.",
        "function_name": "apply_digit_expansion",
        "code": """
            def apply_digit_expansion(input):
                result = []
                prev = None
                for char in input:
                    if char.isdigit():
                        if prev is not None:
                            result.extend([prev] * int(char))
                    else:
                        result.append(char)
                        prev = char
                return "".join(result)
            """
    },
    {
        "id": "private_24",
        "source": "private",
        "task_description": "In a given integers list, replace each item with the nearest earlier item with opposite parity.",
        "function_name": "nearest_opposite_parity",
        "code": """
            def nearest_opposite_parity(vals):
                prev_even = None
                prev_odd = None
                out = []
                for x in vals:
                    if x % 2 == 0:
                        out.append(prev_odd if prev_odd is not None else x)
                        prev_even = x
                    else:
                        out.append(prev_even if prev_even is not None else x)
                        prev_odd = x
                return out
            """
    },
    {
        "id": "private_25",
        "source": "private",
        "task_description": "Given a list of strings, return a new list where each element is the previous string rotated by the length of the current string.",
        "function_name": "shadow_rotate",
        "code": """
            def shadow_rotate(s):
                if res s:
                    return []
                res = [s[0]]
                for i in range(1, len(s)):
                    prev = s[i - 1]
                    if prev == "":
                        res.append("")
                        continue
                    shift = len(s[i]) % len(prev)
                    res.append(prev[-shift:] + prev[:-shift] if shift else prev)
                return res
            """
    },
    {
        "id": "private_26",
        "source": "private",
        "task_description": "Return a sum of character positions, excluding characters at masked indices.",
        "function_name": "sum_char_positions",
        "code": """
            def sum_char_positions(s, masked_indices):
                sum = 0
                masked = set(masked_indices)
                for i, ch in enumerate(s.lower()):
                    if i in masked:
                        continue
                    if 'a' <= ch <= 'z':
                        sum += ord(ch) - ord('a') + 1
                return sum
            """
    },
    {
        "id": "private_27",
        "source": "private",
        "task_description": "Trim n items from both ends of a list and rotate the remaining list left by k positions.",
        "function_name": "trim_and_rotate",
        "code": """
            def trim_and_rotate(vals, n, k):
                if n * 2 >= len(vals):
                    return []
                trimmed = vals[n:len(vals) - n]
                if not trimmed:
                    return []
                k = k % len(trimmed)
                return trimmed[k:] + trimmed[:k]
            """
    },
    {
        "id": "private_28",
        "source": "private",
        "task_description": "Given a string, shift only consonants forward by 1 alphabetically.",
        "function_name": "shift_consonants_forward",
        "code": """
            def shift_consonants_forward(s):
                vowels = set("aeiou")
                def shift(ch):
                    base = ch.lower()
                    if not ("a" <= base <= "z"):
                        return ch
                    if base in vowels:
                        return ch
                    next = chr(((ord(base) - ord("a") + 1) % 26) + ord("a"))
                    if next in vowels:
                        next = chr(((ord(next) - ord("a") + 1) % 26) + ord("a"))
                    return next.upper() if ch.isupper() else next
                return "".join(shift(char) for char in s)
            """
    },
    {
        "id": "private_29",
        "source": "private",
        "task_description": "Given list of ints, return a new list of same length where item i is the count of earlier elements that have at least one common digit with nums[i].",
        "function_name": "prefix_digit_overlap",
        "code": """
            def prefix_digit_overlap(nums):
                seen_digits = []
                res = []
                for num in nums:
                    mask = 0
                    for ch in str(abs(int(num))):
                        mask |= 1 << int(ch)
                    hits = 0
                    for m in seen_digits:
                        if m & mask:
                            hits += 1
                    res.append(hits)
                    seen_digits.append(mask)
                return res
            """
    },
    {
        "id": "private_30",
        "source": "private",
        "task_description": "Compress a list by replacing each strictly-increasing slice with tuple (start_value, slice_length).",
        "function_name": "compress_increasing_runs",
        "code": """
            def compress_increasing_runs(vals):
                if not vals:
                    return []
                res = []
                start = vals[0]
                length = 1
                for i in range(1, len(vals)):
                    if vals[i] > vals[i - 1]:
                        length += 1
                    else:
                        res.append((start, length) if length > 1 else start)
                        start = vals[i]
                        length = 1
                res.append((start, length) if length > 1 else start)
                return res
            """
    },
    {
        "id": "private_31",
        "source": "private",
        "task_description": "Apply right rotation for a given string by the number of distinct chars.",
        "function_name": "rotate_words_by_uniques",
        "code": """
            def rotate_words_by_uniques(s):
                res = []
                for token in s.split():
                    letters = [c.lower() for c in token if c.isalpha()]
                    if not letters:
                        res.append(token)
                        continue
                    k = len(set(letters)) % len(token)
                    res.append(token[-k:] + token[:-k] if k else token)
                return " ".join(res)
            """
    },
    {
        "id": "private_32",
        "source": "private",
        "task_description": "Given a list of strings, return the first index where the cumulative concatenation of previos strings contains any character three or more times.",
        "function_name": "first_triple_char_index",
        "code": """
            def first_triple_char_index(strings):
                counts = {}
                for i, s in enumerate(strings):
                    for ch in s:
                        counts[ch] = counts.get(ch, 0) + 1
                        if counts[ch] >= 3:
                            return i
                return None
            """
    },
    {
        "id": "private_33",
        "source": "private",
        "task_description": "In a given string, replace a character that repeats three consecutivel times with its next ASCII character.",
        "function_name": "triple_to_next_ascii",
        "code": """
            def triple_to_next_ascii(s):
                result = []
                i = 0
                while i < len(s):
                    if i + 2 < len(s) and s[i] == s[i + 1] == s[i + 2]:
                        result.append(chr(ord(s[i]) + 1))
                        i += 3
                    else:
                        result.append(s[i])
                        i += 1
                return "".join(result)
            """
    },
    {
        "id": "private_34",
        "source": "private",
        "task_description": "Return the longest prefix that does not exceed the current maximum by more than a given threshold",
        "function_name": "max_margin_prefix",
        "code": """
            def max_margin_prefix(values, threshold):
                if not values:
                    return []
                current_max = values[0]
                out = [values[0]]
                for val in values[1:]:
                    if val > current_max + threshold:
                        break
                    out.append(val)
                    if val > current_max:
                        current_max = val
                return out
            """
    },
    {
        "id": "private_35",
        "source": "private",
        "task_description": "Compute the sum of absolute differences between items that are k indices apart.",
        "function_name": "absolute_different_sum_k",
        "code": """
            def absolute_different_sum_k(vals, k):
                if k <= 0 or k >= len(vals):
                    return 0
                sum = 0
                for i in range(len(vals) - k):
                    sum += abs(vals[i] - vals[i + k])
                return sum
            """
    },
    { 
        "id": "private_36", 
        "source": "private", 
        "task_description": "Return the factorial of a non-negative integer.", 
        "function_name": "factorial", 
        "code": """ 
            def factorial(n): 
                if n == 0: 
                    return 1 
                return n * factorial(n - 1) 
            """ 
    }, 
    { 
        "id": "private_37", 
        "source": "private", 
        "task_description": "Check whether a number is prime.", 
        "function_name": "is_prime", 
        "code": """ 
            def is_prime(n): 
                if n <= 1: 
                    return False 
                for i in range(2, int(n ** 0.5) + 1): 
                    if n % i == 0: 
                        return False 
                return True 
        """ 
    }, 
    { 
        "id": "private_38", 
        "source": "private", 
        "task_description": "Return the nth Fibonacci number.", 
        "function_name": "fibonacci", 
        "code": """ 
            def fibonacci(n): 
                if n <= 1: 
                    return n 
                a, b = 0, 1 
                for _ in range(n - 1): 
                    a, b = b, a + b 
                return b 
        """ 
    }, 
    { 
        "id": "private_39", 
        "source": "private", 
        "task_description": "Return the sum of all elements in a list.", 
        "function_name": "sum_list", 
        "code": """ 
            def sum_list(nums): 
                return sum(nums) 
        """ 
    }, 
    { 
        "id": "private_40", 
        "source": "private", 
        "task_description": "Return the maximum element in a list.", 
        "function_name": "max_element", 
        "code": """ 
            def max_element(nums): 
                return max(nums) 
        """ 
    }, 
    { 
        "id": "private_41", 
        "source": "private", 
        "task_description": "Check whether a string is a palindrome.", 
        "function_name": "is_palindrome", 
        "code": """ 
            def is_palindrome(text): 
                return text == text[::-1] 
        """ 
    }, 
    { 
        "id": "private_42", 
        "source": "private", 
        "task_description": "Count the number of vowels in a string.", 
        "function_name": "count_vowels", 
        "code": """ 
            def count_vowels(text): 
                return sum(1 for ch in text.lower() if ch in "aeiou") 
        """ 
    }, 
    { 
        "id": "private_43", 
        "source": "private", 
        "task_description": "Return a list with duplicate elements removed.", 
        "function_name": "remove_duplicates", 
        "code": """ 
            def remove_duplicates(lst): 
                return list(set(lst)) 
        """ 
    }, 
    { 
        "id": "private_44", 
        "source": "private", 
        "task_description": "Sort a list in ascending order.", 
        "function_name": "sort_list", 
        "code": """ 
            def sort_list(nums): 
                return sorted(nums) 
        """ 
    }, 
    { 
        "id": "private_45", 
        "source": "private", 
        "task_description": "Return the reverse of a string.", 
        "function_name": "reverse_string", 
        "code": """ 
            def reverse_string(text): 
                return text[::-1] 
        """ 
    }, 
    { 
        "id": "private_46", 
        "source": "private", 
        "task_description": "Find the average value of a list of numbers.", 
        "function_name": "average", 
        "code": """ 
            def average(nums): 
                return sum(nums) / len(nums) 
        """ 
    }, 
    { 
        "id": "private_47", 
        "source": "private", 
        "task_description": "Check whether two strings are anagrams.", 
        "function_name": "are_anagrams", 
        "code": """ 
            def are_anagrams(a, b): 
                return sorted(a) == sorted(b) 
        """ 
    }, 
    { 
        "id": "private_48", 
        "source": "private", 
        "task_description": "Return the number of words in a string.", 
        "function_name": "word_count", 
        "code": """ 
            def word_count(text): 
                return len(text.split()) 
        """ 
    }, 
    { 
        "id": "private_49", 
        "source": "private", 
        "task_description": "Find the minimum element in a list.", 
        "function_name": "min_element", 
        "code": """ 
            def min_element(nums): 
                return min(nums) 
        """ 
    }, 
    { 
        "id": "private_50", 
        "source": "private", 
        "task_description": "Check if a list is sorted in non-decreasing order.", 
        "function_name": "is_sorted", 
        "code": """ 
            def is_sorted(nums): 
                return all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1)) 
        """ 
    }, 
    { 
        "id": "private_51", 
        "source": "private", 
        "task_description": "Return the square of each number in a list.", 
        "function_name": "square_list", 
        "code": """ 
            def square_list(nums): 
                return [x * x for x in nums] 
        """ 
    }, 
    { 
        "id": "private_52", 
        "source": "private", 
        "task_description": "Count occurrences of each character in a string.", 
        "function_name": "char_frequency", 
        "code": """ 
            def char_frequency(text): 
                freq = {} 
                for ch in text: 
                    freq[ch] = freq.get(ch, 0) + 1 
                return freq 
        """ 
    }, 
    { 
        "id": "private_53", 
        "source": "private", 
        "task_description": "Return True if all elements in a list are unique.", 
        "function_name": "all_unique", 
        "code": """ 
            def all_unique(nums): 
                return len(nums) == len(set(nums)) 
        """ 
    }, 
    { 
        "id": "private_54", 
        "source": "private", 
        "task_description": "Merge two lists into one.", 
        "function_name": "merge_lists", 
        "code": """ 
            def merge_lists(a, b): 
                return a + b 
        """ 
    }, 
    { 
        "id": "private_55", 
        "source": "private", 
        "task_description": "Return the index of the maximum element in a list.", 
        "function_name": "index_of_max", 
        "code": """ 
            def index_of_max(nums): 
                return nums.index(max(nums)) 
        """ 
    }, 
    { 
        "id": "private_56", 
        "source": "private", 
        "task_description": "Check if a number is even.", 
        "function_name": "is_even", 
        "code": """ 
            def is_even(n): 
                return n % 2 == 0 
        """ 
    }, 
    { 
        "id": "private_57", 
        "source": "private", 
        "task_description": "Return the absolute difference between two numbers.", 
        "function_name": "absolute_difference", 
        "code": """ 
            def absolute_difference(a, b): 
                return abs(a - b) 
        """ 
    }, 
    { 
        "id": "private_58", 
        "source": "private", 
        "task_description": "Count how many times a value appears in a list.", 
        "function_name": "count_occurrences", 
        "code": """ 
            def count_occurrences(lst, value): 
                return lst.count(value) 
        """ 
    }, 
    { 
        "id": "private_59", 
        "source": "private", 
        "task_description": "Return a list of even numbers from a list.", 
        "function_name": "filter_even", 
        "code": """ 
            def filter_even(nums): 
                return [x for x in nums if x % 2 == 0] 
        """ 
    }, 
    { 
        "id": "private_60", 
        "source": "private", 
        "task_description": "Check whether a key exists in a dictionary.", 
        "function_name": "has_key", 
        "code": """ 
            def has_key(d, key): 
                return key in d 
        """ 
    }, 
    { 
        "id": "private_61", 
        "source": "private", 
        "task_description": "Return the length of the longest word in a string.", 
        "function_name": "longest_word_length", 
        "code": """ 
            def longest_word_length(text): 
                return max(len(w) for w in text.split()) 
        """ 
    }, 
    { 
        "id": "private_62", 
        "source": "private", 
        "task_description": "Return True if all characters in a string are digits.", 
        "function_name": "is_numeric", 
        "code": """ 
            def is_numeric(text): 
                return text.isdigit() 
        """ 
    }, 
    { 
        "id": "private_63", 
        "source": "private", 
        "task_description": "Remove whitespace from both ends of a string.", 
        "function_name": "strip_whitespace", 
        "code": """ 
            def strip_whitespace(text): 
                return text.strip() 
        """ 
    }, 
    { 
        "id": "private_64", 
        "source": "private", 
        "task_description": "Return the product of all numbers in a list.", 
        "function_name": "product", 
        "code": """ 
            def product(nums): 
                result = 1 
                for x in nums: 
                    result *= x 
                return result 
        """ 
    }, 
    { 
        "id": "private_65", 
        "source": "private", 
        "task_description": "Convert all characters in a string to lowercase.", 
        "function_name": "to_lowercase", 
        "code": """ 
            def to_lowercase(text): 
                return text.lower() 
        """ 
    }, 
    { 
        "id": "private_66", 
        "source": "private", 
        "task_description": "Return the last element of a list.", 
        "function_name": "last_element", 
        "code": """ 
            def last_element(lst): 
                return lst[-1] 
        """ 
    }, 
    { 
        "id": "private_67", 
        "source": "private", 
        "task_description": "Check if a string starts with a given prefix.", 
        "function_name": "starts_with", 
        "code": """ 
            def starts_with(text, prefix): 
                return text.startswith(prefix) 
        """ 
    }, 
    { 
        "id": "private_68", 
        "source": "private", 
        "task_description": "Return the count of positive numbers in a list.", 
        "function_name": "count_positive", 
        "code": """ 
            def count_positive(nums): 
                return sum(1 for x in nums if x > 0) 
        """ 
    }, 
    { 
        "id": "private_69", 
        "source": "private", 
        "task_description": "Swap the keys and values of a dictionary.", 
        "function_name": "invert_dict", 
        "code": """ 
            def invert_dict(d): 
                return {v: k for k, v in d.items()} 
        """ 
    }, 
    { 
        "id": "private_70", 
        "source": "private", 
        "task_description": "Check whether a string is a palindrome.", 
        "function_name": "is_palindrome", 
        "code": """ 
            def is_palindrome(text): 
                return text == text[::-1] 
        """ 
    },
    {
        "id": "private_71",
        "source": "private",
        "task_description": "Return True if the list can be made strictly increasing by modifying at most one middle element. You may choose one index i (1..n-2) and replace nums[i] with any integer value v such that nums[i-1] < v < nums[i+1]. Endpoints cannot be changed.",
        "function_name": "fix_strictly_increasing_by_one_clamp",
        "code": """
            def fix_strictly_increasing_by_one_clamp(nums):
                n = len(nums)
                if n <= 2:
                    return True
            
                def is_strict(a):
                    return all(a[i] < a[i+1] for i in range(len(a) - 1))
            
                if is_strict(nums):
                    return True
        
                for i in range(1, n - 1):
                    lo = nums[i - 1] + 1
                    hi = nums[i + 1] - 1
                    if lo > hi:
                        continue

                    v = nums[i]
                    
                    if v < lo:
                        v = lo
                    elif v > hi:
                        v = hi
            
                    candidate = nums[:]
                    candidate[i] = v
                    if is_strict(candidate):
                        return True
            
                return False
        """
    },
    {
        "id": "private_72",
        "source": "private",
        "task_description": "For each element, return the distance to the nearest earlier equal value, or 0 if none.",
        "function_name": "distance_to_prev_equal",
        "code": """
            def distance_to_prev_equal(nums):
                last = {}
                out = []
                for i, x in enumerate(nums):
                    out.append(i - last[x] if x in last else 0)
                    last[x] = i
                return out
        """
    },
    {
        "id": "private_73",
        "source": "private",
        "task_description": "Collapse consecutive characters of the same category inside the string (alpha/digit/space/other), but keep the first and last character unchanged.",
        "function_name": "edge_locked_type_collapse",
        "code": """
            def edge_locked_type_collapse(text):
                if len(text) <= 2:
                    return text
            
                first = text[0]
                last = text[-1]
                mid = text[1:-1]
            
                out = []
                prev_c = None
                for ch in mid:
                    if ch.isalpha():
                        c = "A"
                    elif ch.isdigit():
                        c = "D"
                    elif ch == " ":
                        c = "S"
                    else:
                        c = "O"
            
                    if c != prev_c:
                        out.append(ch)
                        prev_c = c
            
                return first + "".join(out) + last
        """
    },
    {
        "id": "private_74",
        "source": "private",
        "task_description": "Keep only dictionary keys whose value is a local maximum compared to lexicographic neighbor keys.",
        "function_name": "lex_neighbor_peaks",
        "code": """
            def lex_neighbor_peaks(d):
                if not d:
                    return {}
            
                keys = sorted(d.keys())
                out = {}
                for i, k in enumerate(keys):
                    v = d[k]
                    left = d[keys[i - 1]] if i > 0 else None
                    right = d[keys[i + 1]] if i + 1 < len(keys) else None
            
                    if left is None and right is None:
                        out[k] = v
                    elif left is None:
                        if v > right:
                            out[k] = v
                    elif right is None:
                        if v > left:
                            out[k] = v
                    else:
                        if v > left and v > right:
                            out[k] = v
            
                return out
        """
    },
    {
        "id": "private_75",
        "source": "private",
        "task_description": "Split by commas, trim whitespace, ignore empty tokens, and return (token,count) preserving first-seen order.",
        "function_name": "comma_token_counts",
        "code": """
            def comma_token_counts(text):
                items = []
                for raw in text.split(","):
                    t = raw.strip()
                    if t:
                        items.append(t)
            
                counts = {}
                order = []
                for t in items:
                    if t not in counts:
                        counts[t] = 0
                        order.append(t)
                    counts[t] += 1
            
                return [(t, counts[t]) for t in order]
        """
    },
    {
        "id": "private_76",
        "source": "private",
        "task_description": "Return the first index i where distinct(prefix[0..i]) > distinct(suffix[i+1..]). Return None if not found.",
        "function_name": "first_distinct_majority_split",
        "code": """
            def first_distinct_majority_split(vals):
                if not vals:
                    return None
            
                suffix = {}
                for x in vals:
                    suffix[x] = suffix.get(x, 0) + 1
            
                seen = set()
                for i, x in enumerate(vals):
                    seen.add(x)
                    suffix[x] -= 1
                    if suffix[x] == 0:
                        del suffix[x]
                    if len(seen) > len(suffix):
                        return i
            
                return None
        """
    },
    {
        "id": "private_77",
        "source": "private",
        "task_description": "For each space-separated word, rotate its letters right by the number of vowels in that word (non-letters stay).",
        "function_name": "vowel_rotate_words",
        "code": """
            def vowel_rotate_words(sentence):
                vowels = set("aeiouAEIOU")
            
                def rotate_word(w):
                    letters = [c for c in w if c.isalpha()]
                    if not letters:
                        return w
                    k = sum(1 for c in letters if c in vowels) % len(letters)
                    if k == 0:
                        return w
                    rot = letters[-k:] + letters[:-k]
            
                    out = []
                    j = 0
                    for c in w:
                        if c.isalpha():
                            out.append(rot[j].upper() if c.isupper() else rot[j].lower())
                            j += 1
                        else:
                            out.append(c)
                    return "".join(out)
            
                return " ".join(rotate_word(w) for w in sentence.split(" "))
        """
    },
    {
        "id": "private_78",
        "source": "private",
        "task_description": "Return a list where each element i is the number of direction changes in nums[:i+1]. Direction is based on sign of consecutive differences; equal values do not change direction.",
        "function_name": "prefix_direction_changes",
        "code": """
            def prefix_direction_changes(nums):
                if not nums:
                    return []
            
                out = [0]
                changes = 0
                prev_dir = 0  # -1 down, +1 up, 0 unknown/no movement yet
            
                for i in range(1, len(nums)):
                    diff = nums[i] - nums[i - 1]
                    if diff == 0:
                        cur_dir = 0
                    elif diff > 0:
                        cur_dir = 1
                    else:
                        cur_dir = -1
            
                    if cur_dir != 0:
                        if prev_dir != 0 and cur_dir != prev_dir:
                            changes += 1
                        prev_dir = cur_dir
            
                    out.append(changes)
            
                return out
        """
    },
    {
        "id": "private_79",
        "source": "private",
        "task_description": "Return a dict mapping each integer to a list of indices where it appears, but only keep integers that appear at least twice.",
        "function_name": "duplicate_index_map",
        "code": """
            def duplicate_index_map(nums):
                tmp = {}
                for i, x in enumerate(nums):
                    tmp.setdefault(x, []).append(i)
                return {k: v for k, v in tmp.items() if len(v) >= 2}
        """
    },
    {
        "id": "private_80",
        "source": "private",
        "task_description": "Return the longest substring that starts and ends with the same char and contains no digits. If tie, return leftmost.",
        "function_name": "longest_nodigit_same_ends",
        "code": """
            def longest_nodigit_same_ends(s):
                best = ""
                best_i = 0
            
                for i in range(len(s)):
                    if s[i].isdigit():
                        continue
                    for j in range(len(s) - 1, i - 1, -1):
                        if s[j].isdigit():
                            continue
                        if s[i] == s[j]:
                            sub = s[i:j + 1]
                            if any(ch.isdigit() for ch in sub):
                                continue
                            if len(sub) > len(best):
                                best = sub
                                best_i = i
                            break
            
                return best
        """
    },
    {
        "id": "private_81",
        "source": "private",
        "task_description": "Return True if there exists a contiguous subarray of length >=2 whose average is an integer.",
        "function_name": "has_integer_average_window",
        "code": """
            def has_integer_average_window(nums):
                n = len(nums)
                if n < 2:
                    return False
            
                for i in range(n):
                    total = 0
                    for j in range(i, n):
                        total += nums[j]
                        length = j - i + 1
                        if length >= 2 and total % length == 0:
                            return True
            
                return False
        """
    },
    {
        "id": "private_82",
        "source": "private",
        "task_description": "Compare two strings by extending the shorter one by cycling its own characters until it matches the longer length. Return the number of mismatched positions. If one string is empty, mismatches equals the length of the other.",
        "function_name": "cyclic_pad_mismatch",
        "code": """
            def cyclic_pad_mismatch(a, b):
                if a == "" and b == "":
                    return 0
                if a == "":
                    return len(b)
                if b == "":
                    return len(a)
            
                if len(a) >= len(b):
                    long_s, short_s = a, b
                else:
                    long_s, short_s = b, a
            
                m = len(long_s)
                k = len(short_s)
                mismatches = 0
                for i in range(m):
                    if long_s[i] != short_s[i % k]:
                        mismatches += 1
                return mismatches
        """
    },
    {
        "id": "private_83",
        "source": "private",
        "task_description": "For each prefix nums[:i+1], return the current mode (most frequent value). If there is a tie, choose the value whose most recent occurrence is latest. If still tied, choose the smaller value.",
        "function_name": "prefix_mode_recent_tie",
        "code": """
            def prefix_mode_recent_tie(nums):
                counts = {}
                last_pos = {}
                best = None
                best_count = 0
            
                out = []
                for i, x in enumerate(nums):
                    counts[x] = counts.get(x, 0) + 1
                    last_pos[x] = i
            
                    c = counts[x]
                    if best is None:
                        best = x
                        best_count = c
                    else:
                        if c > best_count:
                            best = x
                            best_count = c
                        elif c == best_count:
                            if last_pos[x] > last_pos[best]:
                                best = x
                            elif last_pos[x] == last_pos[best] and x < best:
                                best = x
            
                    out.append(best)
                    
                return out
        """
    },
    {
        "id": "private_84",
        "source": "private",
        "task_description": "Normalize a list of dicts by ensuring every dict has all keys seen; missing keys get default_value.",
        "function_name": "normalize_dict_rows",
        "code": """
            def normalize_dict_rows(rows, default_value=0):
                all_keys = set()
                for r in rows:
                    all_keys.update(r.keys())
            
                out = []
                for r in rows:
                    nr = {}
                    for k in all_keys:
                        nr[k] = r.get(k, default_value)
                    out.append(nr)
            
                return out
        """
    },
    {
        "id": "private_85",
        "source": "private",
        "task_description": "Return True if two strings have the same multiset of letters ignoring case and ignoring non-letters.",
        "function_name": "same_letter_multiset",
        "code": """
            def same_letter_multiset(a, b):
                def freq(s):
                    f = {}
                    for ch in s:
                        if ch.isalpha():
                            c = ch.lower()
                            f[c] = f.get(c, 0) + 1
                    return f
                return freq(a) == freq(b)
        """
    },
    {
        "id": "private_86",
        "source": "private",
        "task_description": "For each element nums[i], return the distance to the nearest earlier element that is coprime with it (gcd==1). If none exists, return 0.",
        "function_name": "nearest_prev_coprime_distance",
        "code": """
            def nearest_prev_coprime_distance(nums):
                out = []
                for i, x in enumerate(nums):
                    dist = 0
                    for j in range(i - 1, -1, -1):
                        a = abs(x)
                        b = abs(nums[j])
                        while b:
                            a, b = b, a % b
                        if a == 1:
                            dist = i - j
                            break
                    out.append(dist)
                return out
        """
    },
    {
        "id": "private_87",
        "source": "private",
        "task_description": "Split a list into chunks so that within each chunk, no adjacent pair (vals[k], vals[k+1]) repeats. When adding the next element would repeat an adjacent pair already seen in the current chunk, start a new chunk at that next element.",
        "function_name": "split_on_repeat_adjacent_pair",
        "code": """
            def split_on_repeat_adjacent_pair(vals):
                if not vals:
                    return []
            
                chunks = []
                cur = [vals[0]]
                seen_pairs = set()
            
                for x in vals[1:]:
                    pair = (cur[-1], x)
                    if pair in seen_pairs:
                        chunks.append(cur)
                        cur = [x]
                        seen_pairs = set()
                    else:
                        seen_pairs.add(pair)
                        cur.append(x)
            
                chunks.append(cur)
                return chunks
        """
    },
    {
        "id": "private_88",
        "source": "private",
        "task_description": "Return the number of distinct 2-character windows in a string (overlapping).",
        "function_name": "count_distinct_bigrams",
        "code": """
    def count_distinct_bigrams(text):
        if len(text) < 2:
            return 0
        return len({text[i:i+2] for i in range(len(text) - 1)})
    """
    },
    {
        "id": "private_89",
        "source": "private",
        "task_description": "For each maximal run of equal values, keep at most (start_index % 3) + 1 elements from that run. Return the filtered list.",
        "function_name": "run_cap_by_position_pattern",
        "code": """
            def run_cap_by_position_pattern(vals):
                if not vals:
                    return []
            
                out = []
                i = 0
                n = len(vals)
                while i < n:
                    start = i
                    x = vals[i]
                    while i < n and vals[i] == x:
                        i += 1
                    run_len = i - start
                    cap = (start % 3) + 1
                    keep = cap if run_len >= cap else run_len
                    out.extend([x] * keep)
            
                return out
        """
    },
    {
        "id": "private_90",
        "source": "private",
        "task_description": "Given a string, replace each digit d with d copies of the previous character; if digit occurs first, keep it unchanged.",
        "function_name": "expand_digits_by_previous_char",
        "code": """
            def expand_digits_by_previous_char(text):
                out = []
                prev = None
                for ch in text:
                    if ch.isdigit():
                        if prev is None:
                            out.append(ch)
                        else:
                            out.extend([prev] * int(ch))
                    else:
                        out.append(ch)
                        prev = ch
                return "".join(out)
        """
    },
    {
        "id": "private_91",
        "source": "private",
        "task_description": "Return the count of indices i where nums[i] is strictly greater than the average of nums[:i]. (Ignore i=0.)",
        "function_name": "count_above_prefix_average",
        "code": """
            def count_above_prefix_average(nums):
                if len(nums) < 2:
                    return 0
            
                total = nums[0]
                count = 0
                for i in range(1, len(nums)):
                    avg = total / i
                    if nums[i] > avg:
                        count += 1
                    total += nums[i]
            
                return count
        """
    },
    {
        "id": "private_92",
        "source": "private",
        "task_description": "For each index i, find the nearest different index to the left and right whose value has gcd(nums[i], nums[j]) > 1. Return the sum of those two neighbor values (missing side counts as 0).",
        "function_name": "sum_nearest_gcd_neighbor",
        "code": """
            def sum_nearest_gcd_neighbor(nums):
                n = len(nums)
                out = []
            
                for i in range(n):
                    left = 0
                    for j in range(i - 1, -1, -1):
                        if nums[j] == nums[i]:
                            continue
                        a = abs(nums[i])
                        b = abs(nums[j])
                        while b:
                            a, b = b, a % b
                        if a > 1:
                            left = nums[j]
                            break
            
                    right = 0
                    for j in range(i + 1, n):
                        if nums[j] == nums[i]:
                            continue
                        a = abs(nums[i])
                        b = abs(nums[j])
                        while b:
                            a, b = b, a % b
                        if a > 1:
                            right = nums[j]
                            break
            
                    out.append(left + right)
            
                return out
        """
    },
    {
        "id": "private_93",
        "source": "private",
        "task_description": "Return True if a list contains a 'mirror pair' (i,j) such that nums[i]==nums[j] and i+j is odd.",
        "function_name": "has_odd_index_sum_equal_pair",
        "code": """
            def has_odd_index_sum_equal_pair(nums):
                seen_even = set()
                seen_odd = set()
            
                for i, x in enumerate(nums):
                    if i % 2 == 0:
                        if x in seen_odd:
                            return True
                        seen_even.add(x)
                    else:
                        if x in seen_even:
                            return True
                        seen_odd.add(x)
            
                return False
        """
    },
    {
        "id": "private_94",
        "source": "private",
        "task_description": "Rotate each word right by (number of distinct letters in the word) mod word length. Non-letters stay in place.",
        "function_name": "rotate_by_unique_letters",
        "code": """
            def rotate_by_unique_letters(text):
                def rotate_token(tok):
                    letters = [c for c in tok if c.isalpha()]
                    if not letters:
                        return tok
            
                    k = len({c.lower() for c in letters}) % len(letters)
                    if k == 0:
                        return tok
            
                    rot = letters[-k:] + letters[:-k]
                    out = []
                    j = 0
                    for c in tok:
                        if c.isalpha():
                            out.append(rot[j])
                            j += 1
                        else:
                            out.append(c)
                    return "".join(out)
            
                return " ".join(rotate_token(t) for t in text.split(" "))
        """
    },
    {
        "id": "private_95",
        "source": "private",
        "task_description": "Given a list of strings, return a list where element i is the count of earlier strings sharing the same first and last character.",
        "function_name": "prefix_suffix_signature_counts",
        "code": """
            def prefix_suffix_signature_counts(words):
                seen = {}
                out = []
                for w in words:
                    if w == "":
                        sig = ("", "")
                    else:
                        sig = (w[0], w[-1])
                    out.append(seen.get(sig, 0))
                    seen[sig] = seen.get(sig, 0) + 1
                return out
        """
    },
    {
        "id": "private_96",
        "source": "private",
        "task_description": "Given a list of integers and modulus m>0, find the shortest contiguous sublist whose set of residues (x mod m) equals the set of residues present in the full list. If tie, return the leftmost window.",
        "function_name": "shortest_window_residue_cover",
        "code": """
            def shortest_window_residue_cover(nums, m):
                if m <= 0:
                    raise ValueError("m must be positive")
                if not nums:
                    return []
            
                target = set([x % m for x in nums])
                best_len = None
                best_i = 0
                best_j = -1
            
                for i in range(len(nums)):
                    seen = set()
                    for j in range(i, len(nums)):
                        seen.add(nums[j] % m)
                        if seen == target:
                            cur_len = j - i + 1
                            if best_len is None or cur_len < best_len:
                                best_len = cur_len
                                best_i, best_j = i, j
                            break
            
                return nums[best_i:best_j + 1] if best_len is not None else []
        """
    },
    {
        "id": "private_97",
        "source": "private",
        "task_description": "Compute total cost from purchases (item,qty,coupon). Apply coupon percentage discounts from coupons dict, capped at 30% per line. Ignore unknown items. None/unknown coupon means no discount.",
        "function_name": "discounted_purchase_total",
        "code": """
            def discounted_purchase_total(prices, purchases, coupons):
                total = 0.0
                for item, qty, coupon in purchases:
                    if item not in prices:
                        continue
            
                    line = prices[item] * qty
                    pct = 0
            
                    if coupon is not None and coupon in coupons:
                        pct = coupons[coupon]
                        if pct > 30:
                            pct = 30
                        elif pct < 0:
                            pct = 0
            
                    total += line * (1 - pct / 100.0)
            
                return total
        """
    },
    {
        "id": "private_98",
        "source": "private",
        "task_description": "For each element nums[i], count how many earlier elements nums[j] satisfy abs(nums[i] - nums[j]) <= radius. Return the list of counts.",
        "function_name": "count_earlier_close_values",
        "code": """
            def count_earlier_close_values(nums, radius=2):
                out = []
                seen = []
                for x in nums:
                    cnt = 0
                    for y in seen:
                        if abs(x - y) <= radius:
                            cnt += 1
                    out.append(cnt)
                    seen.append(x)
                return out
        """
    },
    {
        "id": "private_99",
        "source": "private",
        "task_description": "Given a string, remove the smallest number of characters so that no character appears more than max_count times.",
        "function_name": "limit_char_frequency",
        "code": """
            def limit_char_frequency(text, max_count):
                if max_count < 0:
                    return ""
            
                freq = {}
                out = []
                for ch in text:
                    freq[ch] = freq.get(ch, 0) + 1
                    if freq[ch] <= max_count:
                        out.append(ch)
                return "".join(out)
        """
    },
    {
        "id": "private_100",
        "source": "private",
        "task_description": "Return a list of tuples (value, start_index, length) for each maximal run of equal values.",
        "function_name": "run_segments",
        "code": """
            def run_segments(vals):
                if not vals:
                    return []
            
                out = []
                start = 0
                for i in range(1, len(vals) + 1):
                    if i == len(vals) or vals[i] != vals[i - 1]:
                        out.append((vals[start], start, i - start))
                        start = i
                return out
        """
    },
    {
        "id": "private_101",
        "source": "private",
        "task_description": "Given a list of strings, return those strings whose length is a strict local minimum compared to neighbors (by index).",
        "function_name": "length_local_minima",
        "code": """
            def length_local_minima(words):
                out = []
                for i, w in enumerate(words):
                    left = len(words[i - 1]) if i > 0 else None
                    right = len(words[i + 1]) if i + 1 < len(words) else None
                    cur = len(w)
            
                    if left is None and right is None:
                        out.append(w)
                    elif left is None:
                        if cur < right:
                            out.append(w)
                    elif right is None:
                        if cur < left:
                            out.append(w)
                    else:
                        if cur < left and cur < right:
                            out.append(w)
            
                return out
        """
    },
    {
        "id": "private_102",
        "source": "private",
        "task_description": "Given a list of integers, return True if the multiset of absolute differences between consecutive items contains a duplicate.",
        "function_name": "has_duplicate_adjacent_differences",
        "code": """
            def has_duplicate_adjacent_differences(nums):
                if len(nums) < 3:
                    return False
            
                seen = set()
                for i in range(1, len(nums)):
                    d = abs(nums[i] - nums[i - 1])
                    if d in seen:
                        return True
                    seen.add(d)
            
                return False
        """
    },
    {
        "id": "private_103",
        "source": "private",
        "task_description": "Convert a list of pairs (key,value) into a dict mapping each key to a list of its values in encounter order.",
        "function_name": "group_pairs_to_lists",
        "code": """
            def group_pairs_to_lists(pairs):
                out = {}
                for k, v in pairs:
                    out.setdefault(k, []).append(v)
                return out
        """
    },
    {
        "id": "private_104",
        "source": "private",
        "task_description": "Return the maximum length of a prefix such that the number of vowels seen so far is never less than the number of consonants.",
        "function_name": "max_vowel_dominant_prefix",
        "code": """
            def max_vowel_dominant_prefix(text):
                vowels = set("aeiouAEIOU")
                v = 0
                c = 0
                best = 0
            
                for i, ch in enumerate(text):
                    if ch.isalpha():
                        if ch in vowels:
                            v += 1
                        else:
                            c += 1
                    if v >= c:
                        best = i + 1
            
                return best
        """
    },
    {
        "id": "private_105",
        "source": "private",
        "task_description": "Given a list of integers, return a new list where element i is nums[i] plus the nearest earlier multiple of nums[i] (or 0 if none).",
        "function_name": "add_nearest_prev_multiple",
        "code": """
            def add_nearest_prev_multiple(nums):
                out = []
                for i, x in enumerate(nums):
                    prev = 0
                    if x != 0:
                        for j in range(i - 1, -1, -1):
                            if nums[j] % x == 0:
                                prev = nums[j]
                                break
                    out.append(x + prev)
                return out
        """
    }, 
]