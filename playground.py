import random

your_name = input('Please enter your name:')
my_name = "Bobby"

def sign_off(name):
  print(f"Thank you for the opportunity, {name}!")

def express_gratitude(messages=['Thanks!']):
  return random.choice(messages)

def repeat(message, count):
  for _ in range(count):
    print(message)

gratitude_messages = [
  "Thank you for taking the time to review my application.",
  "I'm excited about the opportunity to contribute to your team.",
  "Your company's vision and values align perfectly with my aspirations.",
  "I'm confident I can make a positive impact in this role.",
  "Thank you for considering my qualifications."
]

gratitude = express_gratitude(gratitude_messages)
print(gratitude)

# Introduce yourself
sign_off(your_name)
print(f"Sincerely,", my_name)