import os
import sys
import logging
from flask import request, jsonify
import requests
import json

# Add parent directory to path to allow importing routes module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Only import and register routes when not running directly
if __name__ != "__main__":
    from routes import app
else:
    # When running directly, create a new Flask app
    from flask import Flask
    app = Flask(__name__)


logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_PROJECT = os.environ.get("OPENAI_PROJECT")

# OpenAI API configuration
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_TEMPERATURE = 0.1
HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

# AI fallback configuration
ENABLE_AI_FALLBACK = True

# Cipher configuration constants
class config:
    DEFAULT_RAILFENCE_RAILS = 3
    DEFAULT_KEYWORD = "SHADOW"
    DEFAULT_CAESAR_SHIFT = 13
    MEANINGFUL_WORDS = ["MISSION", "TARGET", "AGENT", "OPERATION", "SECURE", "CONFIRM", "COMPLETE", "STATUS", "REPORT", "INTEL"]


@app.route('/operation-safeguard', methods=['POST'])
def operation_safeguard():
    """
    Expects JSON like:
    {
      "challenge_one": {...},
      "challenge_two": [[x1, y1], [x2, y2], ...],
      "challenge_three": "log string"
    }
    """
    payload = request.get_json(force=True)

    # Challenge 1 - Reverse Obfuscation Analysis
    challenge_one_data = payload.get("challenge_one", {})
    challenge_one_out = None
    
    if challenge_one_data and "transformations" in challenge_one_data and "transformed_encrypted_word" in challenge_one_data:
        # Define transformation functions
        def mirror_words(x):
            """Reverse each word in the sentence, keeping word order"""
            words = x.split()
            return ' '.join(word[::-1] for word in words)
        
        def encode_mirror_alphabet(x):
            """Replace each letter with its mirror in the alphabet (a ↔️ z, b ↔️ y, ..., A ↔️ Z)"""
            result = ""
            for char in x:
                if char.isalpha():
                    if char.islower():
                        result += chr(ord('z') - (ord(char) - ord('a')))
                    else:
                        result += chr(ord('Z') - (ord(char) - ord('A')))
                else:
                    result += char
            return result
        
        def toggle_case(x):
            """Switch uppercase letters to lowercase and vice versa"""
            return x.swapcase()
        
        def swap_pairs(x):
            """Swap characters in pairs within each word; if odd length, last char stays"""
            words = x.split()
            result_words = []
            for word in words:
                chars = list(word)
                for i in range(0, len(chars) - 1, 2):
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]
                result_words.append(''.join(chars))
            return ' '.join(result_words)
        
        def encode_index_parity(x):
            """Rearrange each word: even indices first, then odd indices"""
            words = x.split()
            result_words = []
            for word in words:
                even_chars = [word[i] for i in range(0, len(word), 2)]
                odd_chars = [word[i] for i in range(1, len(word), 2)]
                result_words.append(''.join(even_chars + odd_chars))
            return ' '.join(result_words)
        
        def double_consonants(x):
            """Double every consonant (letters other than a, e, i, o, u)"""
            vowels = set('aeiouAEIOU')
            result = ""
            for char in x:
                result += char
                if char.isalpha() and char not in vowels:
                    result += char
            return result
        
        # Define reverse transformation functions
        def reverse_mirror_words(x):
            """Reverse of mirror_words"""
            return mirror_words(x)  # Same as forward
        
        def reverse_encode_mirror_alphabet(x):
            """Reverse of encode_mirror_alphabet"""
            return encode_mirror_alphabet(x)  # Same as forward
        
        def reverse_toggle_case(x):
            """Reverse of toggle_case"""
            return toggle_case(x)  # Same as forward
        
        def reverse_swap_pairs(x):
            """Reverse of swap_pairs"""
            return swap_pairs(x)  # Same as forward
        
        def reverse_encode_index_parity(x):
            """Reverse of encode_index_parity"""
            words = x.split()
            result_words = []
            for word in words:
                # Reconstruct original order from even+odd arrangement
                if len(word) <= 1:
                    result_words.append(word)
                else:
                    # Split back into even and odd parts
                    mid = (len(word) + 1) // 2  # Even indices count
                    even_part = word[:mid]
                    odd_part = word[mid:]
                    
                    # Reconstruct original order
                    original = [''] * len(word)
                    even_idx = 0
                    odd_idx = 0
                    
                    for i in range(len(word)):
                        if i % 2 == 0:  # Even index
                            original[i] = even_part[even_idx] if even_idx < len(even_part) else ''
                            even_idx += 1
                        else:  # Odd index
                            original[i] = odd_part[odd_idx] if odd_idx < len(odd_part) else ''
                            odd_idx += 1
                    
                    result_words.append(''.join(original))
            return ' '.join(result_words)
        
        def reverse_double_consonants(x):
            """Reverse of double_consonants"""
            vowels = set('aeiouAEIOU')
            result = ""
            i = 0
            while i < len(x):
                result += x[i]
                # If current char is a consonant and next char is the same, skip the next
                if (i + 1 < len(x) and x[i] == x[i + 1] and 
                    x[i].isalpha() and x[i] not in vowels):
                    i += 1  # Skip the doubled consonant
                i += 1
            return result
        
        # Parse transformations string
        transformations_str = challenge_one_data["transformations"]
        transformed_word = challenge_one_data["transformed_encrypted_word"]
        
        # Extract function names from the transformations string
        import re
        function_matches = re.findall(r'(\w+)\(x\)', transformations_str)
        
        # Create mapping of function names to reverse functions
        reverse_functions = {
            'mirror_words': reverse_mirror_words,
            'encode_mirror_alphabet': reverse_encode_mirror_alphabet,
            'toggle_case': reverse_toggle_case,
            'swap_pairs': reverse_swap_pairs,
            'encode_index_parity': reverse_encode_index_parity,
            'double_consonants': reverse_double_consonants
        }
        
        # Apply reverse transformations in opposite order
        current_word = transformed_word
        for func_name in reversed(function_matches):
            if func_name in reverse_functions:
                current_word = reverse_functions[func_name](current_word)
                logger.info(f"Applied reverse {func_name}: {current_word}")
        
        challenge_one_out = current_word
    else:
        # Fallback to AI-powered analysis if traditional methods fail
        if (ENABLE_AI_FALLBACK and challenge_one_data and 
            ("transformations" in challenge_one_data or "transformed_encrypted_word" in challenge_one_data)):
            try:
                transformations = challenge_one_data.get("transformations", "")
                transformed_word = challenge_one_data.get("transformed_encrypted_word", "")
                
                prompt = f"""
You are a cryptanalysis expert. Analyze the following obfuscated text and reverse the transformations to find the original message.

Transformations applied: {transformations}
Transformed text: {transformed_word}

Common transformation patterns to consider:
1. mirror_words(x) - Reverse each word individually
2. encode_mirror_alphabet(x) - Replace letters with their alphabet mirror (a↔z, b↔y, etc.)
3. toggle_case(x) - Switch case of all letters
4. swap_pairs(x) - Swap characters in pairs within words
5. encode_index_parity(x) - Rearrange: even indices first, then odd indices
6. double_consonants(x) - Double every consonant letter

Apply the reverse transformations in the correct order to recover the original text.
Respond with ONLY the decrypted text, no explanations.
"""
                
                data = {
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a skilled cryptanalyst specializing in text obfuscation reversal."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": OPENAI_TEMPERATURE
                }
                resp = requests.post(OPENAI_URL, headers=HEADERS, data=json.dumps(data))
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"].strip()
                    challenge_one_out = content
                    logger.info("Challenge 1 AI fallback successful: %s", content)
                else:
                    logger.error("Challenge 1 AI fallback failed: %s", resp.text)
                    challenge_one_out = {"error": "AI fallback failed", "details": resp.text}
            except Exception as e:
                logger.error("Challenge 1 AI fallback exception: %s", e)
                challenge_one_out = {"error": "AI fallback exception", "details": str(e)}
        else:
            challenge_one_out = {"error": "Invalid challenge_one data format"}

    # Challenge 2 (coordinate analysis via OpenAI)
    coords = payload.get("challenge_two", [])
    challenge_two_out = None
    if coords:
        coords_text = "\n".join(str(c) for c in coords)
        
        # Enhanced prompt with multiple analysis approaches
        prompt = f"""
You are an expert in pattern recognition and spatial analysis for cybersecurity intelligence.

TASK: Analyze the following coordinate data to extract a meaningful parameter or pattern.

COORDINATE DATA:
{coords_text}

ANALYSIS INSTRUCTIONS:
1. **Spatial Pattern Recognition**: Look for coordinates that form recognizable patterns (letters, numbers, shapes)
2. **Anomaly Detection**: Identify and filter out obvious outliers or decoy coordinates
3. **Clustering Analysis**: Group coordinates that appear to belong together
4. **Dimensional Analysis**: Consider both X and Y relationships
5. **Symbol Recognition**: Try to identify if coordinates form letters, numbers, or symbols

COMMON PATTERNS TO LOOK FOR:
- ASCII art representations of letters/numbers
- Grid-based patterns
- Geometric shapes that might represent symbols
- Sequential patterns that could spell words or numbers

EXAMPLES:
- Coordinates forming the letter "A" might yield parameter: "A" or 1
- Coordinates forming the number "5" might yield parameter: 5
- Coordinates forming a word might yield parameter: "WORD" or a numerical representation

OUTPUT FORMAT:
Respond with a JSON object containing:
- "parameter": The extracted meaningful value (number, letter, or word)
- "pattern_type": The type of pattern identified
- "confidence": Your confidence level (0-100)
- "reasoning": Brief explanation of your analysis

Example response:
{{ "parameter": 5, "pattern_type": "digit", "confidence": 95, "reasoning": "Coordinates form the digit 5 when plotted" }}
"""

        try:
            data = {
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a skilled cryptanalyst."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": OPENAI_TEMPERATURE
            }
            resp = requests.post(OPENAI_URL, headers=HEADERS, data=json.dumps(data))
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                logger.info("Challenge 2 OpenAI response: %s", content)
                challenge_two_out = content
            else:
                logger.error("OpenAI API error: %s", resp.text)
                challenge_two_out = {"error": resp.text}
        except Exception as e:
            logger.error("Challenge 2 OpenAI call failed: %s", e)
            challenge_two_out = {"error": str(e)}

    # Challenge 3 - Operational Intelligence Extraction
    log_string = payload.get("challenge_three", "")
    challenge_three_out = None
    
    if log_string:
        # Parse log entry to extract cipher type and encrypted payload
        import re
        
        # Extract cipher type and encrypted payload from log
        cipher_match = re.search(r'CIPHER_TYPE:\s*(\w+)', log_string)
        payload_match = re.search(r'ENCRYPTED_PAYLOAD:\s*(\w+)', log_string)
        
        if cipher_match and payload_match:
            cipher_type = cipher_match.group(1)
            encrypted_payload = payload_match.group(1)
            
            # Implement cipher decryption functions
            def decrypt_railfence(text, rails=config.DEFAULT_RAILFENCE_RAILS):
                """Decrypt rail fence cipher with 3 rails"""
                if not text:
                    return ""
                
                # Create the rail pattern
                rail_pattern = []
                for i in range(rails):
                    rail_pattern.append([])
                
                # Calculate the pattern for reading
                rail_lengths = [0] * rails
                direction = 1
                rail = 0
                
                for i in range(len(text)):
                    rail_lengths[rail] += 1
                    rail += direction
                    if rail == rails - 1 or rail == 0:
                        direction = -direction
                
                # Fill the rails
                text_index = 0
                for rail in range(rails):
                    for pos in range(rail_lengths[rail]):
                        rail_pattern[rail].append(text[text_index])
                        text_index += 1
                
                # Read the decrypted text
                result = ""
                direction = 1
                rail = 0
                rail_positions = [0] * rails
                
                for i in range(len(text)):
                    result += rail_pattern[rail][rail_positions[rail]]
                    rail_positions[rail] += 1
                    rail += direction
                    if rail == rails - 1 or rail == 0:
                        direction = -direction
                
                return result
            
            def decrypt_keyword(text, keyword=config.DEFAULT_KEYWORD):
                """Decrypt keyword substitution cipher using 'SHADOW'"""
                # Create alphabet with keyword first
                keyword_chars = []
                for char in keyword.upper():
                    if char not in keyword_chars:
                        keyword_chars.append(char)
                
                # Add remaining alphabet characters
                alphabet = keyword_chars + [chr(i) for i in range(ord('A'), ord('Z') + 1) 
                                         if chr(i) not in keyword_chars]
                
                # Create substitution mapping
                substitution = {}
                for i, char in enumerate(alphabet):
                    substitution[char] = chr(ord('A') + i)
                
                # Decrypt the text
                result = ""
                for char in text.upper():
                    if char in substitution:
                        result += substitution[char]
                    else:
                        result += char
                
                return result
            
            def decrypt_polybius(text):
                """Decrypt Polybius square cipher (5x5 grid, I/J combined)"""
                # Create Polybius square
                polybius_square = [
                    ['A', 'B', 'C', 'D', 'E'],
                    ['F', 'G', 'H', 'I', 'K'],  # I and J share position
                    ['L', 'M', 'N', 'O', 'P'],
                    ['Q', 'R', 'S', 'T', 'U'],
                    ['V', 'W', 'X', 'Y', 'Z']
                ]
                
                # Create reverse mapping
                reverse_map = {}
                for row in range(5):
                    for col in range(5):
                        reverse_map[f"{row+1}{col+1}"] = polybius_square[row][col]
                
                # Decrypt pairs of digits
                result = ""
                for i in range(0, len(text), 2):
                    if i + 1 < len(text):
                        pair = text[i:i+2]
                        if pair in reverse_map:
                            result += reverse_map[pair]
                        else:
                            result += pair
                    else:
                        result += text[i]
                
                return result
            
            # Decrypt based on cipher type
            decrypted_text = ""
            if cipher_type == "RAILFENCE":
                decrypted_text = decrypt_railfence(encrypted_payload)
            elif cipher_type == "KEYWORD":
                decrypted_text = decrypt_keyword(encrypted_payload)
            elif cipher_type == "POLYBIUS":
                decrypted_text = decrypt_polybius(encrypted_payload)
            elif cipher_type == "ROTATION_CIPHER":
                # Handle rotation cipher (Caesar cipher)
                def decrypt_rotation(text, shift=13):
                    result = ""
                    for char in text.upper():
                        if char.isalpha():
                            result += chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
                        else:
                            result += char
                    return result
                decrypted_text = decrypt_rotation(encrypted_payload)
            else:
                decrypted_text = f"Unknown cipher type: {cipher_type}"
            
            challenge_three_out = {
                "cipher_type": cipher_type,
                "encrypted_payload": encrypted_payload,
                "decrypted_text": decrypted_text
            }
        else:
            challenge_three_out = {"error": "Could not parse cipher type or encrypted payload from log"}
    else:
        challenge_three_out = {"error": "No challenge_three log string provided"}

    # Challenge 4 - Final Communication Decryption
    challenge_four_data = payload.get("challenge_four", "")
    challenge_four_out = None
    
    if challenge_four_data:
        # First try traditional decryption methods
        traditional_result = None
        # Challenge 4: Use recovered components to decrypt final message
        # This would typically involve combining results from challenges 1, 2, and 3
        # For now, we'll implement a general decryption framework
        
        def decrypt_final_message(encrypted_message, key_components):
            """
            Decrypt the final message using components from previous challenges
            """
            # This is a placeholder implementation
            # In a real scenario, this would use the actual decryption algorithm
            # based on the recovered components from challenges 1, 2, and 3
            
            # For demonstration, we'll implement a simple Vigenère cipher decryption
            def vigenere_decrypt(text, key):
                result = ""
                key_index = 0
                for char in text.upper():
                    if char.isalpha():
                        # Decrypt using Vigenère cipher
                        shift = ord(key[key_index % len(key)].upper()) - ord('A')
                        decrypted_char = chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
                        result += decrypted_char
                        key_index += 1
                    else:
                        result += char
                return result
            
            # Try different decryption methods based on the message format
            if isinstance(challenge_four_data, str):
                # If it's a string, try various decryption methods
                decrypted_message = challenge_four_data
                
                # Method 1: Simple Caesar cipher (shift by 13)
                def caesar_decrypt(text, shift=config.DEFAULT_CAESAR_SHIFT):
                    result = ""
                    for char in text.upper():
                        if char.isalpha():
                            result += chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
                        else:
                            result += char
                    return result
                
                # Method 2: Atbash cipher (A=Z, B=Y, etc.)
                def atbash_decrypt(text):
                    result = ""
                    for char in text.upper():
                        if char.isalpha():
                            result += chr(ord('Z') - (ord(char) - ord('A')))
                        else:
                            result += char
                    return result
                
                # Method 3: Reverse the string
                def reverse_decrypt(text):
                    return text[::-1]
                
                # Try different decryption methods
                decryption_attempts = {
                    "caesar_13": caesar_decrypt(decrypted_message, 13),
                    "caesar_25": caesar_decrypt(decrypted_message, 25),
                    "atbash": atbash_decrypt(decrypted_message),
                    "reverse": reverse_decrypt(decrypted_message),
                    "original": decrypted_message
                }
                
                # Look for meaningful decrypted text
                meaningful_results = []
                for method, result in decryption_attempts.items():
                    # Check if result contains common words or patterns
                    if any(word in result.upper() for word in config.MEANINGFUL_WORDS):
                        meaningful_results.append((method, result))
                
                if meaningful_results:
                    # Return the most likely decrypted message
                    traditional_result = {
                        "decrypted_message": meaningful_results[0][1],
                        "decryption_method": meaningful_results[0][0],
                        "all_attempts": decryption_attempts
                    }
                else:
                    traditional_result = {
                        "decrypted_message": "Unable to decrypt - no meaningful patterns found",
                        "all_attempts": decryption_attempts
                    }
            else:
                traditional_result = {
                    "error": "Challenge four data must be a string message",
                    "received_type": type(challenge_four_data).__name__
                }
        
        # If traditional methods didn't find meaningful results, try AI-powered analysis
        if (ENABLE_AI_FALLBACK and traditional_result and 
            ("Unable to decrypt" in traditional_result.get("decrypted_message", "") or 
             "error" in traditional_result)):
            
            try:
                # Get context from previous challenges for better AI analysis
                context_info = {
                    "challenge_one_result": challenge_one_out,
                    "challenge_two_result": challenge_two_out,
                    "challenge_three_result": challenge_three_out
                }
                
                ai_prompt = f"""
You are an expert cryptanalyst and intelligence analyst. Decrypt the following encrypted message using all available context.

ENCRYPTED MESSAGE: {challenge_four_data}

CONTEXT FROM PREVIOUS CHALLENGES:
- Challenge 1 (Text Obfuscation): {context_info['challenge_one_result']}
- Challenge 2 (Coordinate Analysis): {context_info['challenge_two_result']}
- Challenge 3 (Cipher Decryption): {context_info['challenge_three_result']}

ANALYSIS APPROACH:
1. **Multi-layered Decryption**: Try various cipher types (Caesar, Atbash, Vigenère, substitution, etc.)
2. **Context Integration**: Use information from previous challenges as potential keys or hints
3. **Pattern Recognition**: Look for common words, phrases, or intelligence patterns
4. **Frequency Analysis**: Analyze character frequency for substitution ciphers
5. **Modern Ciphers**: Consider base64, hex, URL encoding, or other modern encoding schemes

COMMON INTELLIGENCE PATTERNS TO LOOK FOR:
- Mission objectives or targets
- Time-sensitive information
- Location data or coordinates
- Agent names or codenames
- Operational procedures
- Status reports or confirmations

OUTPUT FORMAT:
Respond with a JSON object containing:
- "decrypted_message": The decrypted text
- "decryption_method": The method used to decrypt
- "confidence": Confidence level (0-100)
- "reasoning": Brief explanation of your approach
- "key_used": Any key or pattern used for decryption

Example response:
{{ "decrypted_message": "MISSION COMPLETE", "decryption_method": "Caesar cipher", "confidence": 95, "reasoning": "Applied ROT13 to the message", "key_used": "13" }}
"""
                
                data = {
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a master cryptanalyst with expertise in classical and modern ciphers, intelligence analysis, and pattern recognition."},
                        {"role": "user", "content": ai_prompt}
                    ],
                    "temperature": min(OPENAI_TEMPERATURE + 0.1, 0.3)  # Slightly higher for creative analysis
                }
                resp = requests.post(OPENAI_URL, headers=HEADERS, data=json.dumps(data))
                if resp.status_code == 200:
                    ai_content = resp.json()["choices"][0]["message"]["content"].strip()
                    try:
                        # Try to parse as JSON first
                        ai_result = json.loads(ai_content)
                        challenge_four_out = ai_result
                        logger.info("Challenge 4 AI analysis successful: %s", ai_result)
                    except json.JSONDecodeError:
                        # If not JSON, treat as plain text
                        challenge_four_out = {
                            "decrypted_message": ai_content,
                            "decryption_method": "AI_analysis",
                            "confidence": 75,
                            "reasoning": "AI provided plain text response",
                            "fallback_used": True
                        }
                        logger.info("Challenge 4 AI analysis (text): %s", ai_content)
                else:
                    logger.error("Challenge 4 AI analysis failed: %s", resp.text)
                    challenge_four_out = traditional_result
            except Exception as e:
                logger.error("Challenge 4 AI analysis exception: %s", e)
                challenge_four_out = traditional_result
        else:
            challenge_four_out = traditional_result
    else:
        challenge_four_out = {"error": "No challenge_four data provided"}

    return jsonify({
        "challenge_one": challenge_one_out,
        "challenge_two": challenge_two_out,
        "challenge_three": challenge_three_out,
        "challenge_four": challenge_four_out
    })


if __name__ == "__main__":
    # Only run the app if this file is executed directly
    app.run(debug=True, port=5000)
