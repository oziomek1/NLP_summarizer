# -*- coding: utf-8 -*-
import scrapy


def get_category_start_urls():
	return [
        'https://www.sport.pl/sport/0,105641.html?tag=Sport+Extra',
        'https://www.sport.pl/pilka/0,0.html',
        'https://www.sport.pl/pilka/0,65037.html',
        'https://www.sport.pl/igrzyska-olimpijskie/0,0.html',
        'https://www.sport.pl/siatkowka/0,0.html',
        'https://www.sport.pl/lekkoatletyka/0,0.html',
        'https://www.sport.pl/tenis/0,0.html',
        'https://www.sport.pl/koszykowka/0,0.html',
        'https://www.sport.pl/moto/0,0.html',
        'https://www.sport.pl/zimowe/0,0.html',
        'https://www.sport.pl/inne/0,0.html',
        'https://www.sport.pl/sport/0,164252.html',
    ]


class SportSpider(scrapy.Spider):
    name = 'sport'
    allowed_domains = ['sport.pl']
    start_urls = get_category_start_urls()

    custom_settings = {
        'DEPTH_LIMIT': 0
    }

    ARTICLES_URLS = "//article/section[contains(@class, 'body')]/ul/li/header//a/@href"
    NEXT_PAGE_URL = "//footer/div/a[contains(@class, 'next')]/@href"

    TITLE_NODE = "//h1/text()"
    LEAD_NODE = "//div[contains(@id, 'article_lead')]/text()"
    TEXT_NODE = "//div[contains(@id, 'article_body')]/section/p//text() | //div[contains(@id, 'article_body')]/section/h2 | //div[contains(@id, 'article_body')]/div//text() | //div[contains(@id, 'article_body')]/div//h2"

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
