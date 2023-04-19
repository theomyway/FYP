from website import create_app
import matplotlib as mpl

app = create_app()

if __name__ == '__main__':
    mpl.rcParams['agg.path.chunksize'] = 10000
    app.run(debug=True)
