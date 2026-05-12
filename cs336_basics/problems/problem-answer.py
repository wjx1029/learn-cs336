# (a) What Unicode character does chr(0) return?
print(chr(0))
# return a empty string

# (b) How does this character’s string representation (__repr__()) differ from its printed representation?
print(chr(0).__repr__())
# --repr() return hexadecimal number

# (c) What happens when this character occurs in text? It may be helpful to play around with the
# following in your Python interpreter and see if it matches your expectations:
chr(0)
print(chr(0))
"this is a test" + chr(0) + "string"
print("this is a test" + chr(0) + "string")
# return:
# 1. 
# 2.'\x00'
# 3.
# 4.this is a teststring