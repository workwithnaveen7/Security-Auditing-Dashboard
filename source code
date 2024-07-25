import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import mysql.connector
import plotly.express as px
import base64
import io

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

def load_excel_to_mysql(mysql_conn, excel_file, username):
    try:
        xls = pd.ExcelFile(excel_file)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            df = df.where(pd.notnull(df), None)
            cursor = mysql_conn.cursor()
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
            columns = ', '.join([f'`{col}`' for col in df.columns])
            placeholders = ', '.join(['%s'] * len(df.columns))
            insert_query = f"INSERT INTO `{sheet_name}` ({columns}) VALUES ({placeholders})"
            for index, row in df.iterrows():
                row = tuple(str(val)[:255] if isinstance(val, str) else val for val in row)
                cursor.execute(insert_query, row)
            log_query = "INSERT INTO import_logs (username, table_name, import_time) VALUES (%s, %s, NOW())"
            cursor.execute(log_query, (username, sheet_name))
            mysql_conn.commit()
            print(f"Data from sheet '{sheet_name}' imported successfully by {username}")
    except mysql.connector.Error as e:
        print(f"Error importing data to MySQL: {e}")

def retrieve_data_from_tables(mysql_conn, selected_tables, selected_columns):
    try:
        cursor = mysql_conn.cursor()
        data = {}
        for table in selected_tables:
            columns = ', '.join([f"`{col}`" for col in selected_columns.get(table, [])])
            cursor.execute(f"SELECT {columns} FROM `{table}`")
            rows = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            data[table] = pd.DataFrame(rows, columns=column_names)
        return data
    except mysql.connector.Error as e:
        print(f"Error retrieving data from MySQL: {e}")
        return {}

def plot_data(data, graph_types):
    figures = []
    for table, df in data.items():
        if df.empty:
            continue
        for column in df.columns:
            graph_type = graph_types.get(f"{table}:{column}", "bar")
            if df[column].dtype == 'object':  # String data
                counts = df[column].value_counts()
                if graph_type == "bar":
                    fig = px.bar(counts, title=f"Bar Graph for {table} - {column}")
                elif graph_type == "pie":
                    fig = px.pie(names=counts.index, values=counts.values, title=f"Pie Chart for {table} - {column}")
                elif graph_type == "area":
                    fig = px.area(counts.sort_index(), title=f"Area Chart for {table} - {column}")
                else:
                    continue
                figures.append(fig)
                
            else:  # Numeric data
                if graph_type == "bar":
                    fig = px.bar(df[column].sort_values(ascending=False), title=f"Bar Graph for {table} - {column}")
                elif graph_type == "pie":
                    if df[column].nunique() <= 10:  # Example condition
                        fig = px.pie(df[column].value_counts(), title=f"Pie Chart for {table} - {column}")
                    else:
                        continue
                elif graph_type == "area":
                    fig = px.area(df[column].sort_values(ascending=False), title=f"Area Chart for {table} - {column}")
                else:
                    continue
                figures.append(fig)
                
    return figures

app.layout = dbc.Container([
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
        ], width=4)
    ]),
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
                        ), width=8)
                    ]),
                    dbc.Button("Import Data", id="import-data-button", color="primary", className="mr-2"),
                    html.Div(id="import-output")
                ])
            ])
        ], width=8)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Button("Retrieve Data", id="retrieve-data-button", color="primary", className="mr-2"),
                    dcc.Dropdown(id="table-dropdown", multi=True, placeholder="Select tables", style={'width': '100%'}),
                    dcc.Dropdown(id="column-dropdown", multi=True, placeholder="Select columns", style={'width': '100%'}),
                    html.Div(id="column-graph-type-container"),
                    dbc.Button("Plot Graph", id="apply-filters-button", color="secondary", className="mr-2"),
                    html.Div(id="plot-output")
                ])
            ])
        ], width=12)
    ])
], fluid=True)

@app.callback(
    Output('login-output', 'children'),
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
        return "Please enter both username and password."

    print(f"Button clicked: {button_id}")
    print(f"Username: {username}")

    mysql_conn = connect_to_mysql("localhost", "root", "2005", "project_security")
    if not mysql_conn:
        return "Failed to connect to the database."

    create_tables(mysql_conn)

    if button_id == 'register-button':
        success = register_user(mysql_conn, username, password)
        if success:
            return "Registration successful!"
        else:
            return "Registration failed."

    elif button_id == 'login-button':
        success = login_user(mysql_conn, username, password)
        if success:
            return "Login successful!"
        else:
            return "Invalid username or password."


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
    Output('column-graph-type-container', 'children'),
    Input('column-dropdown', 'value')
)
def update_graph_type_dropdowns(selected_columns):
    if not selected_columns:
        return ""

    options = [{'label': graph_type, 'value': graph_type} for graph_type in ["bar", "pie", "area"]]

    dropdowns = []
    for col in selected_columns:
        dropdowns.append(
            dbc.Row([
                dbc.Col(dbc.Label(f"Graph Type for {col}"), width=4),
                dbc.Col(dcc.Dropdown(
                    id={'type': 'graph-type-dropdown', 'index': col},
                    options=options,
                    value='bar',
                    style={'width': '100%'}
                ), width=8)
            ])
        )

    return dropdowns

@app.callback(
    Output('plot-output', 'children'),
    Input('apply-filters-button', 'n_clicks'),
    State('table-dropdown', 'value'),
    State('column-dropdown', 'value'),
    State({'type': 'graph-type-dropdown', 'index': dash.dependencies.ALL}, 'value')
)
def update_plot(n_clicks, selected_tables, selected_columns, graph_types):
    if not n_clicks:
        raise PreventUpdate

    try:
        mysql_conn = connect_to_mysql("localhost", "root", "2005", "project_security")
        if not selected_tables or not selected_columns:
            return "Please select tables and columns."

        columns_by_table = {}
        for col in selected_columns:
            table, column = col.split(':')
            if table not in columns_by_table:
                columns_by_table[table] = []
            columns_by_table[table].append(column)

        data = retrieve_data_from_tables(mysql_conn, list(columns_by_table.keys()), columns_by_table)
        if not data:
            return "No data found."

        graph_types_dict = {col: graph_types[i] for i, col in enumerate(selected_columns)}
        figures = plot_data(data, graph_types_dict)
        return [dcc.Graph(figure=fig) for fig in figures]

    except Exception as e:
        print(f"Error during plot update: {e}")
        return "Error during plot update."
    
    
if __name__ == "__main__":
    app.run_server(debug=True)

