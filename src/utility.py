import PyPDF2
from tqdm import tqdm


# Convert PDF to Text
def pdf_to_text(path):

    file_name = path.split("/")
    book_name = ' '.join(file_name[-1].split('.')[0].split('-')).title()

    pdfFileObj = open(path, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

    res = ''

    if(pdfReader.isEncrypted):
        pdfReader.decrypt('rosebud')

    n = pdfReader.numPages

    print('\n\nReading "' + book_name + '"')
    for i in tqdm(range(n)):
        pageObj = pdfReader.getPage(i)
        res += pageObj.extractText().strip()

    print(len(res.split(' ')))

    return res
