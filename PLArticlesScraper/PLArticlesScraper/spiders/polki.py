# -*- coding: utf-8 -*-
import scrapy


def get_category_start_urls():
	return [
		'https://polki.pl/zdrowie.html',
		'https://polki.pl/dieta-i-fitness.html',
		'https://polki.pl/moda-i-uroda.html',
		'https://polki.pl/dom.html',
		'https://polki.pl/rodzina.html',
	]


class PolkiSpider(scrapy.Spider):
    name = 'polki'
    allowed_domains = ['polki.pl']
    start_urls = get_category_start_urls()

    custom_settings = {
        'DEPTH_LIMIT': 0
    }

    ARTICLES_URLS = "//section[contains(@class, 'slider')]//div[contains(@class, 'inner-box')]//a/@href | //div[contains(@class, 'content-area')]//div[contains(@class, 'list-box')]/div[contains(@class, 'columns')]/a/@href"
    NEXT_PAGE_URL = "//ul[contains(@class, 'pagination')]/li[contains(@class, 'pagination-next')]/a/@href"

    TITLE_NODE = "//div[contains(@class, 'header')]//h1/text()"
    LEAD_NODE = "//div[contains(@class, 'lead')]/text()"
    TEXT_NODE = "//div[contains(@class, 'content-area')]/article/p[not(contains(@class, 'healthNotice'))]//text() | //div[contains(@class, 'content-area')]/article/ul/li//text()  | //div[contains(@class, 'content-area')]/article/h2"

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
            'text': response.xpath(self.TEXT_NODE).extract()[:-1],
        }