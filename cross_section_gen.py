import math

def generate_numbers():
    with open("src/cross_section.txt", "w") as file:
        for i in range(10000):
            number = (math.cos(i/400) + 2) * pow(i, 0.8) / 10000000
            file.write(f"{number} {number}\n")

if __name__ == "__main__":
    generate_numbers()