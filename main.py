from bs4 import BeautifulSoup
import streamlit as st
import pypandoc
import os
import subprocess

# 确保 'pdf_files' 文件夹存在
if not os.path.exists('pdf_files'):
    os.makedirs('pdf_files')

def save_uploadedfile(uploadedfile):
    with open(os.path.join("pdf_files", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return os.path.join("pdf_files", uploadedfile.name)

def pdf_to_html(filepath):
    html_filepath = filepath.rsplit('.', 1)[0] + '.html'
    # 将 Windows 路径转换为 Docker 中的路径
    docker_filepath = filepath.replace("\\", "/").replace("D:", "/mnt/d")
    docker_html_filepath = html_filepath.replace("\\", "/").replace("D:", "/mnt/d")
    # 使用 docker run 命令来运行 pdf2htmlEX
    subprocess.check_call(['docker', 'run', '-v', f'{os.getcwd()}:/pdf', 'bwits/pdf2htmlex', 'pdf2htmlEX', '--zoom', '1.3', docker_filepath, docker_html_filepath])
    return html_filepath

def html_to_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        contents = f.read()

    soup = BeautifulSoup(contents, 'lxml')
    text = soup.get_text()

    with open(filepath.rsplit('.', 1)[0] + '.md', 'w', encoding='utf-8') as f:
        f.write(text)


def main():
    st.title("PDF Reader and Mind Map Generator")

    # Create a two-column layout
    cols = st.columns(2)

    # Left column: PDF import and display
    with cols[0]:
        st.header("PDF Import and Display")
        uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf', 'docx', 'txt'])
        if uploaded_file is not None:
            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
            st.write(file_details)
            file_path = save_uploadedfile(uploaded_file)
            html_file_path = pdf_to_html(file_path)
            html_to_text(html_file_path)
            st.write('文件已成功转换为 markdown 格式')
            # Display the PDF file
            st.markdown(f'<iframe src="{file_path}" width="700" height="800"></iframe>', unsafe_allow_html=True)

            # Save the PDF and Markdown files
            with open(os.path.join("pdf_files", "saved_pdf.pdf"), "wb") as f:
                f.write(uploaded_file.getbuffer())
            with open("output.md", "r", encoding="utf-8") as f_in:
                with open(os.path.join("pdf_files", "saved_md.md"), "w", encoding="utf-8") as f_out:
                    f_out.write(f_in.read())

    # Right column: Card generation and mind map
    with cols[1]:
        # Top half: Card generation
        st.header("Card Generation")
        # TODO: Add code to generate cards

        # Bottom half: Mind map
        st.header("Mind Map")
        # TODO: Add code to generate mind map

if __name__ == "__main__":
    main()



