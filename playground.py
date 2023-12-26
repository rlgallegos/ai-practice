from docx import Document

efforts = [
    'Direct Space',
    'Indirect Space',
    'Strong Weight',
    'Light Weight',
    'Sudden Time',
    'Sustained Time',
    'Bound Flow',
    'Free Flow'
]

actions = [
    'Float',
    'Punch',
    'Glide',
    'Slash',
    'Dab',
    'Wring',
    'Flick',
    'Press'
]

result = []

for effort in efforts:
    result.append('///')
    for action in actions:
        combo = effort + ' - ' + action
        result.append(combo)

for r in result:
    print()
    print(r)

document = Document()

# Add a heading for the list
document.add_heading("Johnny's List", level=1)

# Create a paragraph and add the list items as bullet points
i = 0
for item in result:
    if item == '///':
        document.add_heading(efforts[i // 8])
        paragraph = document.add_paragraph()
    elif i % 2 == 0:
        paragraph.add_run(item + "\u00a0\u00a0\u00a0").font.name = "Times New Roman"
    else:
        paragraph.add_run(item + "\n").font.name = "Times New Roman"
    i += 1

document.save("johnny-list.docx")