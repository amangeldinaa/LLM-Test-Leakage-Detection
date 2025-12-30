PRIVATE_FUNCTIONS = [
    { 
        "id": "private_1", 
        "source": "private", 
        "task_description": "Reverse words whose length is odd.", 
        "function_name": "reverse_odd_words", 
        "code": """\ 
            def reverse_odd_words(words): 
                return [w[::-1] if len(w) % 2 else w for w in words] 
            """ 
    }, 
    { 
        "id": "private_2", 
        "source": "private", 
        "task_description": "Add even-indexed numbers and subtract odd-indexed ones.", 
        "function_name": "alternating_sum", 
        "code": """\ 
            def alternating_sum(nums): 
                return sum(nums[::2]) - sum(nums[1::2]) 
            """ 
    }, 
    { 
        "id": "private_3", 
        "source": "private", 
        "task_description": "Wrap each word in the text with a given marker.", 
        "function_name": "wrap_text", 
        "code": """\ 
            def wrap_text(text, marker): 
                return " ".join(marker + w + marker for w in text.split()) 
            """ 
    }, 
    { 
        "id": "private_4", 
        "source": "private", 
        "task_description": "Count how many times the sign of numbers changes in a list.", 
        "function_name": "count_switch_points", 
        "code": """\ 
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
        "code": """\ 
            def duplicate_last_n(lst, n): 
                return lst + lst[-n:] 
            """ 
    }, 
    { 
        "id": "private_6", 
        "source": "private", 
        "task_description": "Merge two lists by selecting the element with shorter string length at each index.", 
        "function_name": "merge_by_length", 
        "code": """\ 
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
        "code": """\ 
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
        "code": """\ 
            def conditional_multiply(nums, threshold): 
                return [x * x if x > threshold else x for x in nums] 
            """ 
    }, 
    { 
        "id": "private_9", 
        "source": "private", 
        "task_description": "Flatten all list values of a dictionary into a single list.", 
        "function_name": "flatten_dict_values", 
        "code": """\ 
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
        "code": """\ 
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
        "code": """\ 
            def extract_even_positions(text): 
                return "".join(text[i] for i in range(0, len(text), 2)) 
            """ 
    }, 
    { 
        "id": "private_12", 
        "source": "private", 
        "task_description": "Rotate a list to the right by k positions.", 
        "function_name": "rotate_right", 
        "code": """\ 
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
        "code": """\ 
            def replace_if_prefix(words, prefix, token): 
                return [token if w.startswith(prefix) else w for w in words] 
            """ 
    }, 
    { 
        "id": "private_14", 
        "source": "private", 
        "task_description": "Return sorted unique characters from a string.", 
        "function_name": "unique_sorted_chars", 
        "code": """\ 
            def unique_sorted_chars(text): 
                return "".join(sorted(set(text))) 
            """ 
    }, 
    { 
        "id": "private_15", 
        "source": "private", 
        "task_description": "Map each key through a dictionary with a fallback default value.", 
        "function_name": "map_with_default", 
        "code": """\ 
            def map_with_default(keys, mapping, default): 
                return [mapping.get(k, default) for k in keys] 
            """ 
    }, 
    { 
        "id": "private_16", 
        "source": "private", 
        "task_description": "Summarize consecutive repeated values as (value, count).", 
        "function_name": "summarize_runs", 
        "code": """\ 
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
        "code": """\ 
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
        "code": """\ 
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
        "code": """\ 
            def split_on_marker(text, marker): 
                return text.split(marker) 
            """ 
    }, 
    { 
        "id": "private_20", 
        "source": "private", 
        "task_description": "Return the absolute difference if it is below a limit, otherwise return the limit.", 
        "function_name": "bounded_difference", 
        "code": """\ 
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
        "code": """\
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
        "code": """\
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
        "code": """\
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
        "code": """\
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
        "code": """\
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
        "code": """\
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
        "code": """\
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
        "code": """\
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
        "code": """\
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
        "code": """\
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
        "code": """\
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
        "code": """\
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
        "code": """\
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
        "code": """\
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
        "code": """\
            def absolute_different_sum_k(vals, k):
                if k <= 0 or k >= len(vals):
                    return 0
                sum = 0
                for i in range(len(vals) - k):
                    sum += abs(vals[i] - vals[i + k])
                return sum
            """
    },
]