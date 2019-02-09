import numpy as np
import flask
from flask import request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def cube_root(n):
    """Find the cube root of a number
    """
    return n ** (1./3.)


def find_l_a_b(x, y, z, X_n, Y_n, Z_n):
    """Calculates the values of L*,a*,b*
    """
    l_star = (116 * cube_root(y/Y_n)) - 16
    a_star = 500 * (cube_root(x/X_n)-cube_root(y/Y_n))
    b_star = 200 * (cube_root(y/Y_n)-cube_root(z/Z_n))
    return (l_star, a_star, b_star)


def find_delE(l_star, a_star, b_star, l0_star, a0_star, b0_star):
    """Find the value of delta E
    """
    del_l = l0_star - l_star
    del_a = a0_star - a_star
    del_b = b0_star - b_star
    del_E = (del_l**2 + del_a**2 + del_b**2)**0.5
    return del_E


def train_model_predict(x_train, y_train, del_E_input):
    """Train the LinearRegression Model
    """
    x_train_processed = np.array(x_train).reshape(-1, 1)
    y_train_processed = np.array(y_train).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(x_train_processed, y_train_processed)
    return lr.predict(np.array([del_E_input]).reshape(-1, 1))[0][0]


app = flask.Flask(__name__)


@app.route('/calculateph', methods=['POST'])
def calculate_ph_value():
    print(request.json['R'])
    R_in = request.json['R']
    G_in = request.json['G']
    B_in = request.json['B']
    X_n = request.json['X_n']
    Y_n = request.json['Y_n']
    Z_n = request.json['Z_n']
    l0_star = request.json['l0_star']
    a0_star = request.json['a0_star']
    b0_star = request.json['b0_star']
    rgb_values = [(195, 151, 178), (184, 130, 166), (179, 136, 163),
                  (171, 148, 166), (153, 139, 156), (171, 163, 178)]
    x = []
    y = [2, 4, 6, 8, 10, 12]
    for (r, g, b) in rgb_values:
        l_star, a_star, b_star = find_l_a_b(r, g, b, X_n, Y_n, Z_n)
        del_E = find_delE(l_star, a_star, b_star, l0_star, a0_star, b0_star)
        x.append(del_E)
    print(x)
    l_input, a_input, b_input = find_l_a_b(R_in, G_in, B_in, X_n, Y_n, Z_n)
    del_E_input = find_delE(l_input, a_input, b_input,
                            l0_star, a0_star, b0_star)
    output_pH = train_model_predict(x, y, del_E_input)
    return jsonify({
        'pH': output_pH
    })


if __name__ == "__main__":
    app.run(debug=True)
