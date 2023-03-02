import click 


@click.command()
def hello():
    click.echo('hello there')

if __name__ == '__main__':
    hello()