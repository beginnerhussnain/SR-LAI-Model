import sys
import inspect

# Repository root
sys.path.insert(
    0,
    r"D:\Code Playground\VS Code Repository\SR_LAI Model\pypro4sail"
)

# Import the actual pypro4sail.py module
from pypro4sail import pypro4sail as p4s


print("=" * 60)
print("Actual pyPro4Sail module")
print("=" * 60)

print("Module location:")
print(p4s.__file__)

print("\nFunctions and objects:")
print(dir(p4s))


print("\n" + "=" * 60)
print("RUN FUNCTION")
print("=" * 60)

if hasattr(p4s, "run"):
    print("run() exists!")

    print("\nFunction signature:")
    print(inspect.signature(p4s.run))

    print("\nDocumentation:")
    print(inspect.getdoc(p4s.run))

else:
    print("run() does NOT exist in this version.")