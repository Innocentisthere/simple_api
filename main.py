from typing import Annotated, List

import xml.etree.ElementTree as ET
import json
from fastapi import FastAPI, Depends, UploadFile, File
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

app = FastAPI()

engine = create_async_engine('sqlite+aiosqlite:///tokens.db')

new_session = async_sessionmaker(engine, expire_on_commit=False)

tree = ET.parse('example.xml')

async def get_session():
    async with new_session() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


class Base(DeclarativeBase):
    pass


class TokenModel(Base):  # Модель бд
    __tablename__ = 'tokens'
    token: Mapped[int] = mapped_column(primary_key=True)
    page_count: Mapped[int]


class TokenSchema(BaseModel):  # Модель пользователя
    token: int
    page_count: int


class RecognizeRequest(BaseModel):
    model_type: str
    recognition_type: str
    content: List[str]
    user_token: int
    response_type: str


@app.post('/setup_db')
async def setup_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    return {'status': 'success'}


@app.post("/recognize")
async def recognize(request: RecognizeRequest, session: SessionDep):
    model_type = request.model_type
    recognition_type = request.recognition_type
    images = request.content
    user_token = request.user_token
    response_type = request.response_type

    # Model processing
    try:
        root = tree.getroot()
    except ET.ParseError:
        return None
    detected_pages = len(root.findall('.//node[@type="RIL_PAGE"]'))

    user = await session.get(TokenModel, user_token)
    user_pages = user.page_count
    print(user_pages)

    if user is None:
        return {'status': 'error', 'message': 'Token not found'}
    elif user_pages < detected_pages:
        return {'status': 'error', 'message': 'Pages limit exceeded'}
    else:
        extracted_data = []
        page_count = 0
        image_count = 0

        for page_element in root.findall('.//node[@type="RIL_PAGE"]'):
            page_count += 1
            page_width = int(page_element.get('W'))
            page_height = int(page_element.get('H'))
            page_data = {"text": [], "tables": [],
                         "width": page_width, "height": page_height}


            for line_element in page_element.findall('.//node[@type="RIL_TEXTLINE"]'):
                line_text = ""
                for word_element in line_element.findall('.//node[@type="RIL_WORD"]'):
                    if word_element.text:
                        line_text += word_element.text + ' '
                page_data["text"].append(line_text.strip())


            for table_element in page_element.findall('.//node[@type="RIL_TABLE"]'):
                table_data = []
                rows = {}  

                for cell_element in table_element.findall('.//node[@type="RIL_TABLE_CELL"]'):
                    coords = cell_element.get('coords')
                    if coords:
                        col, row = coords.split('_')
                        col = int(col)
                        row = int(row[:-1])
                        if cell_element.text:
                            cell_text = cell_element.text
                        else:
                            cell_text = ""

                        if row not in rows:
                            rows[row] = {}
                        rows[row][col] = cell_text

                sorted_rows = sorted(rows.items())
                for row_index, row_data in sorted_rows:
                    row = [row_data.get(col_index, "")
                           for col_index in sorted(row_data.keys())]
                    table_data.append(row)
                page_data["tables"].append(table_data)

            extracted_data.append(page_data)

            image_count_on_page = len(
                page_element.findall(".//node[@type='RIL_IMAGE']"))
            image_count += image_count_on_page

        metadata = {"page_count": page_count, "image_count": image_count}
        result = {"data": page_data, "metadata": metadata}
        return json.dumps(result, indent=2, ensure_ascii=False)


@app.post('/pay')  # Новый пользователь
async def add_token(data: TokenSchema, session: SessionDep):
    given_token = data.token
    existing_token = await session.execute(select(TokenModel).filter_by(token=given_token))
    token_instance = existing_token.scalars().first()

    if token_instance:
        return {'status': 'token_already_exists', 'token': given_token}
    else:
        new_token = TokenModel(
            token=given_token,
            page_count=data.page_count
        )
        session.add(new_token)
        await session.commit()
        return {'status': 'success'}


@app.put('/pay')  # Добавление страниц существующему пользователю
async def update_token_page_count(data: TokenSchema, session: SessionDep):
    given_token = data.token
    token_to_update = await session.get(TokenModel, given_token)
    if token_to_update is None:
        return {'status': 'error', 'message': 'Token not found'}

    token_to_update.page_count += data.page_count
    await session.commit()
    return {
        'status': 'success',
        'updated_token': {
            'token': given_token,
            'page_count': token_to_update.page_count
        }
    }
