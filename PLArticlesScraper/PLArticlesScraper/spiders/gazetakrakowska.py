# -*- coding: utf-8 -*-
import scrapy


def get_category_start_urls():
	return [
		'http://gazetakrakowska.pl/',
		'https://gazetakrakowska.pl/wiadomosci/',
		'https://gazetakrakowska.pl/motofakty/',
		'https://gazetakrakowska.pl/turystyka/',
		'https://gazetakrakowska.pl/sport/',
		'https://gazetakrakowska.pl/zdrowie/',
		'https://gazetakrakowska.pl/strona-kobiet/'
	]


class GazetakrakowskaSpider(scrapy.Spider):
    name = 'gazetakrakowska'
    allowed_domains = ['gazetakrakowska.pl']
    start_urls = get_category_start_urls()

    custom_settings = {
        'DEPTH_LIMIT': 0
    }

    ARTICLES_URLS = "//article[contains(@class, 'elementRegionalne') and not(contains(@class, 'contentStream'))]/a/@href"
    NEXT_PAGE_URL = "//div[contains(@class, 'stronicowanie')]/a/@href"

    TITLE_NODE = "//header//h1/text()"
    LEAD_NODE = "//div[contains(@class, 'componentsArticleLead')]/text() | //article/div/strong/text()"
    TEXT_NODE = "//article[contains(@class, 'Art')]/div/p/text() | //article[contains(@class, 'Art')]/div/p/a/text()"

    def parse(self, response):
        # iterate through links to articles
        for quote in response.xpath(self.ARTICLES_URLS):
            yield response.follow(quote, callback=self.parse_article)
            
        #iterate through next page links
        for next_page in response.xpath(self.NEXT_PAGE_URL):
            yield response.follow(next_page, self.parse)
            
    def parse_article(self, response):
        yield {
            'url': response.url,
            'title': response.xpath(self.TITLE_NODE).extract(),
            'lead': response.xpath(self.LEAD_NODE).extract(),
            'text': response.xpath(self.TEXT_NODE).extract(),
        }