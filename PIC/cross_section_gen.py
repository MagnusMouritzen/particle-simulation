import math

def generate_numbers():
    with open("src/cross_section.txt", "w") as file:
        for i in range(10000):
            number = (math.sin(pow(i, 2.6) / 50000000) + 1.01) * pow(i, 0.1)
            file.write(f"{number} {number}\n")

if __name__ == "__main__":
    generate_numbers()