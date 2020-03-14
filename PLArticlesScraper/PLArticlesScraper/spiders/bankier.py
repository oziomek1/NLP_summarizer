# -*- coding: utf-8 -*-
import scrapy


class BankierSpider(scrapy.Spider):
    name = 'bankier'
    allowed_domains = ['bankier.pl']
    start_urls = ['http://bankier.pl/']

    custom_settings = {
        'DEPTH_LIMIT': 0
    }

    ARTICLES_URLS = "//div[contains(@id, 'boxDayNews')]/div[contains(@class, 'boxContent')]/ul/li/a[not(@class)]/@href"
    NEXT_PAGE_URL = "//div[contains(@class, 'boxHeader')]/ul/li/a/@href"

    NEWS_ARTICLES_URLS = "//div[contains(@id, 'pageMainContainer')]//div[contains(@class, 'article')]/div/a/@href"
    NEWS_NEXT_PAGE_URLS = "//div[contains(@id, 'pageMainContainer')]//div[contains(@class, 'top')]/a/@href"

    TITLE_NODE = "//article/header/h1/text()"
    LEAD_NODE = "//article/div/p/span[contains(@class, 'lead')]/text()"
    TEXT_NODE = "//article/div/p/text()"
    TEXT_MAIN_POINTS = "//article/div/p/span[not(contains(@class, 'lead'))]"

    def parse(self, response):
        # iterate through links to articles
        for quote in response.xpath(self.ARTICLES_URLS) + response.xpath(self.NEWS_ARTICLES_URLS):
            yield response.follow(quote, callback=self.parse_article)

        #iterate through next page links
        for next_page in response.xpath(self.NEXT_PAGE_URL) + response.xpath(self.NEWS_NEXT_PAGE_URLS):
            yield response.follow(next_page, self.parse)
            
    def parse_article(self, response):
        yield {
            'url': response.url,
            'title': response.xpath(self.TITLE_NODE).extract(),
            'lead': response.xpath(self.LEAD_NODE).extract(),
            'text': response.xpath(self.TEXT_NODE).extract(),
            'text_main_points': response.xpath(self.TEXT_MAIN_POINTS).extract(),
        }