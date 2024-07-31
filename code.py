import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import mysql.connector
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import base64
import io
from wordcloud import WordCloud
import matplotlib.pyplot as plt


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def connect_to_mysql(host, user, password, database):
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            passwd=password,
            database=database
        )
        print("Connected to MySQL database")
        return conn
    except mysql.connector.Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

def create_tables(connection):
    create_users_table_query = """
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(255) NOT NULL UNIQUE,
        password VARCHAR(255) NOT NULL
    )"""
    create_import_logs_table_query = """
    CREATE TABLE IF NOT EXISTS import_logs (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(255) NOT NULL,
        table_name VARCHAR(255) NOT NULL,
        data_hash VARCHAR(255) NOT NULL,
        import_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )"""
    try:
        cursor = connection.cursor()
        cursor.execute(create_users_table_query)
        cursor.execute(create_import_logs_table_query)
        connection.commit()
        print("Tables 'users' and 'import_logs' created successfully")
    except mysql.connector.Error as e:
        print(f"Error creating tables: {e}")



def register_user(connection, username, password):
    insert_query = """
    INSERT INTO users (username, password) VALUES (%s, %s)"""
    try:
        cursor = connection.cursor()
        cursor.execute(insert_query, (username, password))
        connection.commit()
        print(f"User '{username}' registered successfully")
        return True
    except mysql.connector.Error as e:
        print(f"Error registering user: {e}")
        return False

def login_user(connection, username, password):
    select_query = """
    SELECT * FROM users WHERE username = %s AND password = %s"""
    try:
        cursor = connection.cursor()
        cursor.execute(select_query, (username, password))
        user = cursor.fetchone()
        if user:
            print(f"Login successful. Welcome, {username}!")
            return True
        else:
            print("Invalid username or password")
            return False
    except mysql.connector.Error as e:
        print(f"Error authenticating user: {e}")
        return False


def generate_data_hash(df):
    """Generate a hash for the given DataFrame."""
    data_string = df.to_csv(index=False)
    return hashlib.md5(data_string.encode()).hexdigest()

def load_excel_to_mysql(mysql_conn, excel_file, username):
    try:
        xls = pd.ExcelFile(excel_file)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            df = df.where(pd.notnull(df), None)

            data_hash = generate_data_hash(df)

            cursor = mysql_conn.cursor()
            # Check if this data has already been imported
            cursor.execute(
                "SELECT * FROM import_logs WHERE table_name = %s AND data_hash = %s",
                (sheet_name, data_hash)
            )
            if cursor.fetchone():
                print(f"Data from sheet '{sheet_name}' has already been imported.")
                continue

            # Create table if it doesn't exist
            create_table_query = f"CREATE TABLE IF NOT EXISTS `{sheet_name}` ("
            for col in df.columns:
                dtype = 'VARCHAR(255)'
                if pd.api.types.is_integer_dtype(df[col]):
                    dtype = 'INT'
                elif pd.api.types.is_float_dtype(df[col]):
                    dtype = 'FLOAT'
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    dtype = 'DATETIME'
                create_table_query += f"`{col}` {dtype}, "
            create_table_query = create_table_query.rstrip(', ') + ")"
            cursor.execute(create_table_query)

            # Insert data into the table
            columns = ', '.join([f'`{col}`' for col in df.columns])
            placeholders = ', '.join(['%s'] * len(df.columns))
            insert_query = f"INSERT INTO `{sheet_name}` ({columns}) VALUES ({placeholders})"
            for index, row in df.iterrows():
                row = tuple(str(val)[:255] if isinstance(val, str) else val for val in row)
                cursor.execute(insert_query, row)

            # Log the import
            log_query = "INSERT INTO import_logs (username, table_name, data_hash, import_time) VALUES (%s, %s, %s, NOW())"
            cursor.execute(log_query, (username, sheet_name, data_hash))

            mysql_conn.commit()
            print(f"Data from sheet '{sheet_name}' imported successfully by {username}")
    except mysql.connector.Error as e:
        print(f"Error importing data to MySQL: {e}")

def retrieve_data_from_tables(mysql_conn, selected_tables, selected_columns, filters):
    try:
        cursor = mysql_conn.cursor()
        data = {}
        for table in selected_tables:
            columns = ', '.join([f"`{col}`" for col in selected_columns.get(table, [])])
            query = f"SELECT {columns} FROM `{table}`"
            filter_clauses = []
            filter_values = []
            for col, val in filters.items():
                if val:
                    filter_clauses.append(f"`{col.split(':')[1]}` = %s")
                    filter_values.append(val)
            if filter_clauses:
                query += " WHERE " + " AND ".join(filter_clauses)
            print(f"Executing query: {query}")
            print(f"With values: {filter_values}")
            cursor.execute(query, filter_values)
            rows = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            data[table] = pd.DataFrame(rows, columns=column_names)
        return data
    except mysql.connector.Error as e:
        print(f"Error retrieving data from MySQL: {e}")
        return {}


import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np
from PIL import Image
import pandas as pd
from collections import Counter
import io

import plotly.graph_objects as go
import pandas as pd

import plotly.graph_objects as go
import pandas as pd
from collections import defaultdict

def plot_data(data, graph_types):
    figures = []
    for table, df in data.items():
        if df.empty:
            continue
        for column in df.columns:
            graph_type = graph_types.get(f"{table}:{column}", "bar")

            if graph_type == "sankey":
                if len(df.columns) >= 2:
                    # Collect unique nodes and create index mapping
                    node_labels = []
                    links = defaultdict(lambda: {'source': [], 'target': [], 'value': []})
                    node_map = {}

                    # Generate links between consecutive columns
                    for i in range(len(df.columns) - 1):
                        source_col = df.columns[i]
                        target_col = df.columns[i + 1]

                        for _, row in df.iterrows():
                            source_node = row[source_col]
                            target_node = row[target_col]

                            if source_node not in node_map:
                                node_map[source_node] = len(node_labels)
                                node_labels.append(source_node)

                            if target_node not in node_map:
                                node_map[target_node] = len(node_labels)
                                node_labels.append(target_node)

                            source_index = node_map[source_node]
                            target_index = node_map[target_node]

                            links[(source_col, target_col)]['source'].append(source_index)
                            links[(source_col, target_col)]['target'].append(target_index)
                            links[(source_col, target_col)]['value'].append(1)  # Default value

                    # Prepare Sankey diagram data
                    all_sources = []
                    all_targets = []
                    all_values = []

                    for link in links.values():
                        all_sources.extend(link['source'])
                        all_targets.extend(link['target'])
                        all_values.extend(link['value'])

                    fig = go.Figure(go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color='black', width=1.5),
                            label=node_labels,
                        ),
                        link=dict(
                            source=all_sources,
                            target=all_targets,
                            value=all_values
                        )
                    ))
                    fig.update_layout(
                        title_text=f"Sankey Diagram for {table}",
                        height=800,
                        width=1300,
                        font=dict(size=14)  # Adjust the font size here
                    )
                    figures.append(fig)
            else:
                # Existing handling for other graph types
                if df[column].dtype == 'object':  # String data
                    combined_text = " ".join(df[column].dropna().astype(str).tolist())
                    if graph_type == "bar":
                        counts = pd.Series(df[column].dropna().astype(str)).value_counts()
                        fig = px.bar(
                            counts,
                            title=f"Bar Graph for {table} - {column}",
                            labels={'index': column, 'value': 'Count'},
                            color=counts.index,
                            text=counts.values
                        )
                        fig.update_traces(textposition='outside')
                        fig.update_layout(showlegend=False, height=650, width=1200)
                    elif graph_type == "pie":
                        counts = pd.Series(df[column].dropna().astype(str)).value_counts()
                        if not counts.empty:
                            fig = px.pie(names=counts.index, values=counts.values, title=f"Pie Chart for {table} - {column}")
                            figures.append(fig)
                    elif graph_type == "line":
                        counts = pd.Series(df[column].dropna().astype(str)). value_counts().sort_index()
                        fig = px.line(counts, title=f"Line Chart for {table} - {column}")
                    elif graph_type == "wordcloud":
                        word_freq = Counter(df[column].dropna().astype(str))
                        top_words = dict(word_freq.most_common(15))  # Limit to top 15 words

                        # Modify word frequencies to handle multiline text
                        modified_top_words = {}
                        for word, freq in top_words.items():
                            # Only use the first two words if the text contains multiple lines
                            short_word = ' '.join(word.split()[:2])
                            if short_word in modified_top_words:
                                modified_top_words[short_word] += freq
                            else:
                                modified_top_words[short_word] = freq

                        wordcloud = WordCloud(width=1500, height=1300, background_color='white').generate_from_frequencies(modified_top_words)
                        
                        # Save word cloud image to a file
                        img = wordcloud.to_image()
                        img.save("wordcloud.png")
                        
                        # Load image into Plotly
                        with open("wordcloud.png", "rb") as img_file:
                            img_bytes = img_file.read()
                        img_str = "data:image/png;base64," + base64.b64encode(img_bytes).decode('utf-8')
                        
                        # Create Plotly figure with image
                        fig = go.Figure()
                        fig.add_trace(go.Image(source=img_str))
                        fig.update_layout(title=f"Word Cloud for {table} - {column}", xaxis_visible=False, yaxis_visible=False)
                    else:
                        continue
                    figures.append(fig)
                else:  # Numeric data
                    if graph_type == "bar":
                        fig = px.bar(
                            df,
                            x=df.index,
                            y=column,
                            title=f"Bar Graph for {table} - {column}",
                            labels={'index': 'Index', column: 'Value'},
                            color=df.index,
                            text=column
                        )
                        fig.update_traces(textposition='outside')
                        fig.update_layout(showlegend=False, height=600, width=800)
                    elif graph_type == "pie":
                        if df[column].nunique() <= 10:
                            fig = px.pie(df, names=df.index, values=column, title=f"Pie Chart for {table} - {column}")
                            figures.append(fig)
                    elif graph_type == "line":
                        fig = px.line(df, x=df.index, y=column, title=f"Line Chart for {table} - {column}")
                    else:
                        continue
                    figures.append(fig)
    return figures



app.layout = dbc.Container([
    dcc.Store(id='login-state', data={'logged_in': False}),
    dbc.Row([
        dbc.Col(html.H1("Security Auditing Dashboard", className="text-center"), className="mb-5 mt-5")
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dbc.Label("Username"), width=4),
                        dbc.Col(dbc.Input(id="username", type="text", placeholder="Enter username"), width=8)
                    ]),
                    dbc.Row([
                        dbc.Col(dbc.Label("Password"), width=4),
                        dbc.Col(dbc.Input(id="password", type="password", placeholder="Enter password"), width=8)
                    ]),
                    dbc.Button("Register", id="register-button", color="primary", className="mr-2"),
                    dbc.Button("Login", id="login-button", color="secondary"),
                    html.Div(id="login-output")
                ])
            ])
        ], width=15)
    ]),
    html.Div(id='main-content', style={'display': 'none'}, children=[
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(dbc.Label("Excel File"), width=4),
                            dbc.Col(dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Files')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                                multiple=False
                            ), width=15)
                        ]),
                        html.Div(id='upload-output')
                    ])
                ])
            ], width=15)
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(dbc.Label("Select Table"), width=4),
                            dbc.Col(dcc.Dropdown(id='table-dropdown', multi=True), width=8)
                        ]),
                        dbc.Row([
                            dbc.Col(dbc.Label("Select Columns"), width=4),
                            dbc.Col(dcc.Dropdown(id='column-dropdown', multi=True), width=8)
                        ]),
                        html.Div(id='filter-inputs-container'),
                        html.Div(id='graph-type-dropdowns-container'),
                        dbc.Button("Retrieve Data", id="retrieve-data-button", color="primary", className="mr-2"),
                        dbc.Button("Plot Graph", id="apply-filters-button", color="secondary"),
                        html.Div(id='plot-output')
                    ])
                ])
            ], width=12)
        ])
    ])
])


@app.callback(
    Output('upload-output', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('username', 'value')
)
def handle_file_upload(contents, filename, username):
    if contents is None:
        return "No file uploaded."
    
    if not username:
        return "Please login first."

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        excel_file = io.BytesIO(decoded)
        mysql_conn = connect_to_mysql("localhost", "root", "2005", "project_security")
        if mysql_conn:
            create_tables(mysql_conn)
        load_excel_to_mysql(mysql_conn, excel_file, username)
        return f"File '{filename}' uploaded and data imported successfully."
    except Exception as e:
        print(f"Error handling file upload: {e}")
        return f"Error uploading file: {e}"

@app.callback(
    [Output('login-output', 'children'),
     Output('login-state', 'data'),
     Output('main-content', 'style')],
    [Input('register-button', 'n_clicks'),
     Input('login-button', 'n_clicks')],
    [State('username', 'value'),
     State('password', 'value')]
)
def authenticate(register_clicks, login_clicks, username, password):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if not username or not password:
        return "Please enter both username and password.", dash.no_update, {'display': 'none'}

    mysql_conn = connect_to_mysql("localhost", "root", "2005", "project_security")
    if not mysql_conn:
        return "Failed to connect to the database.", dash.no_update, {'display': 'none'}

    create_tables(mysql_conn)

    if button_id == 'register-button':
        success = register_user(mysql_conn, username, password)
        if success:
            return "Registration successful!", {'logged_in': True}, {'display': 'block'}
        else:
            return "Registration failed.", dash.no_update, {'display': 'none'}

    elif button_id == 'login-button':
        success = login_user(mysql_conn, username, password)
        if success:
            return "Login successful!", {'logged_in': True}, {'display': 'block'}
        else:
            return "Invalid username or password.", dash.no_update, {'display': 'none'}


@app.callback(
    [Output('table-dropdown', 'options'),
     Output('column-dropdown', 'options')],
    Input('retrieve-data-button', 'n_clicks'),
    State('table-dropdown', 'value')
)
def update_dropdowns(n_clicks, selected_tables):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    if ctx.triggered[0]['prop_id'] == 'retrieve-data-button.n_clicks':
        try:
            mysql_conn = connect_to_mysql("localhost", "root", "2005", "project_security")
            cursor = mysql_conn.cursor()
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            table_options = [{'label': table[0], 'value': table[0]} for table in tables]

            column_options = []
            if selected_tables:
                for table in selected_tables:
                    cursor.execute(f"SHOW COLUMNS FROM `{table}`")
                    columns = cursor.fetchall()
                    for column in columns:
                        column_options.append({'label': f"{table}: {column[0]}", 'value': f"{table}:{column[0]}"})

            return table_options, column_options

        except mysql.connector.Error as e:
            print(f"Error updating dropdowns: {e}")
            return [], []

@app.callback(
    Output('filter-inputs-container', 'children'),
    Input('column-dropdown', 'value')
)
def update_filter_inputs(selected_columns):
    if not selected_columns:
        return ""

    # Create a list to hold the rows
    filter_inputs = []
    
    # Split selected columns into rows of a fixed number of columns
    num_columns_per_row = 3
    for i in range(0, len(selected_columns), num_columns_per_row):
        row_columns = selected_columns[i:i + num_columns_per_row]
        row_filters = [
            dbc.Col([
                #dbc.Label(f"Filter for {col}"),
                dbc.Input(id={'type': 'filter-input', 'index': col}, type='text', placeholder=f"Enter filter for {col}")
            ], width=4) for col in row_columns
        ]
        filter_inputs.append(dbc.Row(row_filters, className="mb-2"))

    return filter_inputs



@app.callback(
    Output('plot-output', 'children'),
    Input('apply-filters-button', 'n_clicks'),
    State('table-dropdown', 'value'),
    State('column-dropdown', 'value'),
    State({'type': 'filter-input', 'index': dash.dependencies.ALL}, 'value'),
    State({'type': 'filter-input', 'index': dash.dependencies.ALL}, 'id'),
    State({'type': 'graph-type-dropdown', 'index': dash.dependencies.ALL}, 'value'),
    State({'type': 'graph-type-dropdown', 'index': dash.dependencies.ALL}, 'id')
)
def update_plot(n_clicks, selected_tables, selected_columns, filter_values, filter_ids, graph_types, graph_type_ids):
    if not n_clicks:
        raise PreventUpdate

    try:
        mysql_conn = connect_to_mysql("localhost", "root", "2005", "project_security")
        if not selected_tables or not selected_columns:
            return "Please select tables and columns."

        columns_by_table = {}
        filters = {}
        for col in selected_columns:
            table, column = col.split(':')
            if table not in columns_by_table:
                columns_by_table[table] = []
            columns_by_table[table].append(column)

        for i, id_dict in enumerate(filter_ids):
            filters[id_dict['index']] = filter_values[i]

        graph_types_dict = {}
        for i, id_dict in enumerate(graph_type_ids):
            graph_types_dict[id_dict['index']] = graph_types[i]

        data = retrieve_data_from_tables(mysql_conn, list(columns_by_table.keys()), columns_by_table, filters)
        if not data:
            return "No data found."

        figures = plot_data(data, graph_types_dict)

        graphs_and_dropdowns = []
        for col in selected_columns:
            table, column = col.split(':')
            graph_type = graph_types_dict.get(f"{table}:{column}", 'bar')

            dropdown = dcc.Dropdown(
                id={'type': 'graph-type-dropdown', 'index': col},
                options=[
                    {'label': 'Bar Graph', 'value': 'bar'},
                    {'label': 'Pie Chart', 'value': 'pie'},
                    {'label': 'Line Chart', 'value': 'line'},
                    {'label': 'WordCloud Graph', 'value': 'wordcloud'},
                    {'label': 'Sankey Diagram', 'value': 'sankey'}  # Add Sankey option here
                ] ,
                value=graph_type
            )

            graphs_and_dropdowns.append(html.Div([
                dropdown,
                dcc.Graph(figure=figures.pop(0))  # Pop the first figure for this column
            ]))

        return html.Div(graphs_and_dropdowns)
    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == '__main__':
    app.run_server(debug=True)
# Establish the MySQL connection at the start of the application
    mysql_conn = connect_to_mysql("localhost", "root", "2005", "project_security")
    if mysql_conn:
        create_tables(mysql_conn)

