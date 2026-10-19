import unittest
from PIL import Image
from collector import analyse, find_film_links, find_still_urls, dhash, hamming, valid_url


class ColourTests(unittest.TestCase):
    def test_blue_passes(self):
        image = Image.new('RGB', (800, 450), (25, 110, 235))
        result = analyse(image)
        self.assertEqual(result['reason'], 'accepted')
        self.assertEqual(result['colour'], 'blue')
        self.assertGreater(result['majority'], .95)

    def test_grey_fails(self):
        self.assertEqual(analyse(Image.new('RGB', (800, 450), (125, 125, 125)))['reason'], 'too_grey_or_desaturated')

    def test_brown_fails(self):
        self.assertEqual(analyse(Image.new('RGB', (800, 450), (115, 75, 38)))['reason'], 'too_brown')

    def test_rainbow_fails(self):
        im = Image.new('RGB', (800, 450))
        for x in range(im.width):
            for y in range(im.height):
                color = ((200, 15, 25), (10, 135, 215), (20, 180, 35), (210, 185, 20))[x // 200]
                im.putpixel((x, y), color)
        self.assertNotEqual(analyse(im)['reason'], 'accepted')

    def test_html_discovery(self):
        page = '<a href="/works/chihiro/">Film</a><a href="/works/ponyo/">Film2</a>'
        self.assertEqual(set(find_film_links(page)), {'chihiro', 'ponyo'})
        stills = '<a href="https://www.ghibli.jp/gallery/chihiro001.jpg">image</a><img data-src="/gallery/chihiro002.jpg"><img src="https://evil.example/gallery/fake.jpg">'
        self.assertEqual(len(find_still_urls(stills, 'https://www.ghibli.jp/works/chihiro/')), 2)
        self.assertFalse(valid_url('https://evil.example/gallery/chihiro001.jpg'))

    def test_dhash(self):
        one = Image.new('RGB', (800, 450), (11, 150, 230))
        self.assertEqual(hamming(dhash(one), dhash(one)), 0)

if __name__ == '__main__':
    unittest.main()

class IntegrationTests(unittest.TestCase):
    def test_integration_fake_official_gallery(self):
        import argparse
        import io
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from collector import run
        def jpeg(rgb):
            out = io.BytesIO()
            Image.new('RGB', (800, 450), rgb).save(out, 'JPEG')
            return out.getvalue()
        sources = {
            'https://www.ghibli.jp/works/': b'<a href="/works/chihiro/">film</a>',
            'https://www.ghibli.jp/works/chihiro/': (
                b'<a href="/gallery/chihiro001.jpg">one</a>'
                b'<a href="/gallery/chihiro002.jpg">two</a>'
                b'<a href="/gallery/chihiro003.jpg">three</a>'
            ),
            'https://www.ghibli.jp/gallery/chihiro001.jpg': jpeg((25, 110, 235)),
            'https://www.ghibli.jp/gallery/chihiro002.jpg': jpeg((20, 190, 40)),
            'https://www.ghibli.jp/gallery/chihiro003.jpg': jpeg((130, 130, 130)),
        }
        class FakeHTTP:
            def __init__(self, delay): pass
            def get(self, url, maximum): return sources[url]
        with tempfile.TemporaryDirectory() as folder, patch('collector.PoliteHttp', FakeHTTP):
            options = argparse.Namespace(root=Path(folder),delay=2.5,
                discover_hours=24,max_process=30,max_new=12,
                majority=.60,max_neutral=.30,max_brown=.20,
                min_saturation=.30,max_entropy=1.70,dhash_distance=5)
            first = run(options)
            self.assertEqual(first['accepted'], 2)
            self.assertEqual(first['rejected'], 1)
            self.assertTrue((Path(folder)/'by_colour/blue/_contact_sheet.jpg').exists())
            self.assertTrue((Path(folder)/'by_colour/green/_contact_sheet.jpg').exists())
            second = run(options)
            self.assertEqual(second['processed'], 0)
            self.assertTrue((Path(folder)/'manifest.csv').exists())