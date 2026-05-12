# (a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
# UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
# input strings.
# Deliverable: A one-to-two sentence response.

test_string = "hello! こんにちは"
utf8_encoded = test_string.encode("utf-8")
print(utf8_encoded)
utf16_encoded = test_string.encode("utf-16")
print(utf16_encoded)
utf32_encoded = test_string.encode("utf-32")
print(utf32_encoded)
# UTF-8 uses only 256 token IDs and works seamlessly with any input, while UTF-16/32 would require tens of thousands to millions of token IDs, making them impractical and inefficient for neural tokenizers.


# (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string
# into a Unicode string. Why is this function incorrect? Provide an example of an input byte
# string that yields incorrect results.
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
print(decode_utf8_bytes_to_str_wrong("hello!".encode("utf-8")))
# "hello! こんにちは!" will produce incorrect output, beacuse this function cannot decode a character which have two bytes or above

# Give a two-byte sequence that does not decode to any Unicode character(s)

# "\xc2\x20" 0xC2 是一个两字节字符的起始字节，要求第二个字节必须是 0x80–0xBF 范围的继续字节，但空格 0x20 不在这个范围内，因此整个序列无法解码成任何有效的 Unicode 字符